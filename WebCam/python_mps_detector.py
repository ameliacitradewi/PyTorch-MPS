#!/usr/bin/env python3
"""
Python sidecar detector for Swift app.
- Input: newline-delimited JSON on stdin with base64 JPEG frames.
- Output: newline-delimited JSON detections on stdout.
Uses YOLO .pt directly with PyTorch MPS when available.
"""

import base64
import contextlib
import json
import os
import sys
import time
import traceback

def log_stderr(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def choose_device(torch_module, preferred: str) -> str:
    preferred = (preferred or "").strip().lower()
    if preferred in {"mps", "cpu"}:
        if preferred == "mps" and not torch_module.backends.mps.is_available():
            log_stderr("Requested device mps, but MPS is unavailable. Falling back to cpu.")
            return "cpu"
        return preferred

    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_names(names_obj):
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, list):
        return {i: str(name) for i, name in enumerate(names_obj)}
    return {}


def select_target_ids(names_map, keyword_text: str):
    keywords = [x.strip().lower() for x in keyword_text.split(",") if x.strip()]
    if not keywords:
        keywords = ["person", "human", "body", "face", "head", "hand", "arm", "leg", "foot"]

    target_ids = []
    for class_id, class_name in names_map.items():
        lower_name = class_name.lower()
        if any(keyword in lower_name for keyword in keywords):
            target_ids.append(class_id)

    # Keep class 0 when it is person-like in COCO-style models.
    if 0 in names_map and "person" in names_map[0].lower():
        target_ids.append(0)

    return sorted(set(target_ids))


def emit_json(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def extract_color_signature(frame, x1, y1, x2, y2, cv2_module, np_module):
    ix1 = max(0, int(x1))
    iy1 = max(0, int(y1))
    ix2 = min(frame.shape[1], int(x2))
    iy2 = min(frame.shape[0], int(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    roi = frame[iy1:iy2, ix1:ix2]
    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]
    margin_x = int(roi_w * 0.15)
    margin_y = int(roi_h * 0.15)
    if roi_w - (2 * margin_x) > 3 and roi_h - (2 * margin_y) > 3:
        roi = roi[margin_y:roi_h - margin_y, margin_x:roi_w - margin_x]

    sampled = roi[::2, ::2]
    if sampled.size == 0:
        sampled = roi

    hsv = cv2_module.cvtColor(sampled, cv2_module.COLOR_BGR2HSV)
    flat = hsv.reshape(-1, 3)
    return np_module.median(flat, axis=0).astype(np_module.float32)


def hsv_distance(hsv_a, hsv_b):
    hue_delta = abs(float(hsv_a[0]) - float(hsv_b[0]))
    hue_delta = min(hue_delta, 180.0 - hue_delta)
    sat_delta = abs(float(hsv_a[1]) - float(hsv_b[1]))
    val_delta = abs(float(hsv_a[2]) - float(hsv_b[2]))
    weighted_h = hue_delta * 2.2
    weighted_s = sat_delta * 1.0
    weighted_v = val_delta * 0.45
    return (weighted_h * weighted_h + weighted_s * weighted_s + weighted_v * weighted_v) ** 0.5


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    yolo_config_dir = os.environ.get("YOLO_CONFIG_DIR", "").strip()
    if yolo_config_dir:
        os.makedirs(yolo_config_dir, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = yolo_config_dir

    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
    import cv2  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    model_path = os.environ.get("YOLO_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError("YOLO_MODEL_PATH is empty.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    conf_default = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.5"))
    preferred_device = os.environ.get("YOLO_DEVICE", "mps").strip().lower()
    keyword_text = os.environ.get(
        "YOLO_TARGET_KEYWORDS",
        "person,human,body,face,head,hand,arm,leg,foot",
    )

    log_stderr(f"Loading model: {model_path}")
    with contextlib.redirect_stdout(sys.stderr):
        model = YOLO(model_path)
    device = choose_device(torch, preferred_device)
    model.to(device)
    names_map = normalize_names(model.names)
    if not names_map:
        log_stderr("Warning: model.names could not be normalized; class filtering disabled.")
    target_ids = select_target_ids(names_map, keyword_text)
    log_stderr(f"Device: {device}")
    log_stderr(f"Target class IDs: {target_ids if target_ids else 'all'}")
    emit_json(
        {
            "ready": True,
            "device": device,
            "mps_available": bool(torch.backends.mps.is_available()),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
        }
    )

    perf_window_count = 0
    perf_decode_total_ms = 0.0
    perf_infer_total_ms = 0.0
    perf_post_total_ms = 0.0
    perf_total_total_ms = 0.0
    perf_log_interval = 20

    recent_track_window_frames = 45
    profile_ttl_frames = 1200
    max_track_profiles = 256
    recent_color_match_threshold = 85.0
    reid_color_match_threshold = 62.0
    center_match_threshold = 0.28
    center_distance_weight = 55.0
    reid_age_penalty = 0.02
    color_ema = 0.2
    center_ema = 0.3

    track_state = {}
    next_track_id = 1
    frame_seq = 0

    for raw_line in sys.stdin.buffer:
        if not raw_line:
            break
        line = raw_line.strip()
        if not line:
            continue

        frame_id = -1
        try:
            frame_start = time.perf_counter()
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Input JSON must be an object.")
            frame_id = int(request.get("frame_id", -1))
            conf = float(request.get("conf", conf_default))
            jpeg_b64 = request.get("jpeg_b64")
            if not jpeg_b64:
                raise ValueError("jpeg_b64 is missing.")

            jpeg_bytes = base64.b64decode(jpeg_b64)
            frame_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("Failed to decode JPEG frame.")
            decode_done = time.perf_counter()
            frame_seq += 1

            height, width = frame.shape[:2]
            with contextlib.redirect_stdout(sys.stderr):
                result = model(
                    frame,
                    conf=conf,
                    classes=target_ids if target_ids else None,
                    device=device,
                    verbose=False,
                )[0]
            infer_done = time.perf_counter()

            detections = []
            boxes = result.boxes
            used_track_ids = set()
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls.item()) if box.cls is not None else -1
                    label = names_map.get(cls_id, f"class_{cls_id}")
                    confidence = float(box.conf.item()) if box.conf is not None else 0.0
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x1 = max(0.0, min(float(x1), float(width)))
                    x2 = max(0.0, min(float(x2), float(width)))
                    y1 = max(0.0, min(float(y1), float(height)))
                    y2 = max(0.0, min(float(y2), float(height)))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    color_signature = extract_color_signature(frame, x1, y1, x2, y2, cv2, np)
                    center = np.array(
                        [((x1 + x2) * 0.5) / width, ((y1 + y2) * 0.5) / height],
                        dtype=np.float32,
                    )

                    selected_track_id = None
                    best_score = float("inf")
                    if color_signature is not None:
                        # Pass 1: short-gap tracking with color + spatial continuity.
                        for track_id, track in track_state.items():
                            if track_id in used_track_ids:
                                continue
                            age = frame_seq - track["last_seen"]
                            if age > recent_track_window_frames:
                                continue

                            color_distance = hsv_distance(color_signature, track["hsv"])
                            if color_distance > recent_color_match_threshold:
                                continue

                            center_distance = float(np.linalg.norm(center - track["center"]))
                            if center_distance > center_match_threshold:
                                continue

                            score = color_distance + (center_distance * center_distance_weight)
                            if score < best_score:
                                best_score = score
                                selected_track_id = track_id

                        # Pass 2: long-gap re-identification based on clothing color.
                        if selected_track_id is None:
                            for track_id, track in track_state.items():
                                if track_id in used_track_ids:
                                    continue
                                age = frame_seq - track["last_seen"]
                                if age > profile_ttl_frames:
                                    continue

                                color_distance = hsv_distance(color_signature, track["hsv"])
                                if color_distance > reid_color_match_threshold:
                                    continue

                                score = color_distance + (age * reid_age_penalty)
                                if score < best_score:
                                    best_score = score
                                    selected_track_id = track_id

                    if selected_track_id is None:
                        selected_track_id = next_track_id
                        next_track_id += 1
                        track_state[selected_track_id] = {
                            "hsv": color_signature if color_signature is not None else np.zeros(3, dtype=np.float32),
                            "center": center,
                            "last_seen": frame_seq,
                            "seen_count": 1,
                        }
                    else:
                        track = track_state[selected_track_id]
                        if color_signature is not None:
                            track["hsv"] = ((1.0 - color_ema) * track["hsv"]) + (color_ema * color_signature)
                        track["center"] = ((1.0 - center_ema) * track["center"]) + (center_ema * center)
                        track["last_seen"] = frame_seq
                        track["seen_count"] = int(track.get("seen_count", 1)) + 1

                    used_track_ids.add(selected_track_id)
                    detections.append(
                        {
                            "label": label,
                            "track_id": selected_track_id,
                            "confidence": confidence,
                            "x": x1 / width,
                            "y": y1 / height,
                            "w": (x2 - x1) / width,
                            "h": (y2 - y1) / height,
                        }
                    )
            post_done = time.perf_counter()

            stale_track_ids = [
                track_id
                for track_id, track in track_state.items()
                if frame_seq - track["last_seen"] > profile_ttl_frames
            ]
            for stale_track_id in stale_track_ids:
                del track_state[stale_track_id]
            if len(track_state) > max_track_profiles:
                overflow = len(track_state) - max_track_profiles
                oldest = sorted(track_state.items(), key=lambda item: item[1]["last_seen"])[:overflow]
                for track_id, _ in oldest:
                    del track_state[track_id]

            decode_ms = (decode_done - frame_start) * 1000.0
            infer_ms = (infer_done - decode_done) * 1000.0
            post_ms = (post_done - infer_done) * 1000.0
            total_ms = (post_done - frame_start) * 1000.0

            perf_window_count += 1
            perf_decode_total_ms += decode_ms
            perf_infer_total_ms += infer_ms
            perf_post_total_ms += post_ms
            perf_total_total_ms += total_ms

            if perf_window_count >= perf_log_interval:
                log_stderr(
                    "[PyPerf] avg over "
                    f"{perf_window_count} frames | decode {perf_decode_total_ms / perf_window_count:.1f} ms"
                    f" | infer {perf_infer_total_ms / perf_window_count:.1f} ms"
                    f" | post {perf_post_total_ms / perf_window_count:.1f} ms"
                    f" | total {perf_total_total_ms / perf_window_count:.1f} ms"
                )
                perf_window_count = 0
                perf_decode_total_ms = 0.0
                perf_infer_total_ms = 0.0
                perf_post_total_ms = 0.0
                perf_total_total_ms = 0.0

            emit_json({"frame_id": frame_id, "detections": detections})

        except Exception as exc:  # pylint: disable=broad-except
            message = f"{type(exc).__name__}: {exc}"
            log_stderr(message)
            emit_json({"frame_id": frame_id, "detections": [], "error": message})


if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_error:  # pylint: disable=broad-except
        log_stderr("Fatal detector error: " + str(fatal_error))
        log_stderr(traceback.format_exc())
        emit_json({"frame_id": -1, "detections": [], "error": str(fatal_error)})
        sys.exit(1)
