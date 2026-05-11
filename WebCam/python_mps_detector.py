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

    # Always include class 0 if it is person-like in COCO-style models.
    if 0 in names_map and "person" in names_map[0].lower() and 0 not in target_ids:
        target_ids.append(0)

    return sorted(set(target_ids))


def emit_json(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


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

    for raw_line in sys.stdin.buffer:
        if not raw_line:
            break
        line = raw_line.strip()
        if not line:
            continue

        frame_id = None
        try:
            frame_start = time.perf_counter()
            request = json.loads(line)
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

                    detections.append(
                        {
                            "label": label,
                            "confidence": confidence,
                            "x": x1 / width,
                            "y": y1 / height,
                            "w": (x2 - x1) / width,
                            "h": (y2 - y1) / height,
                        }
                    )
            post_done = time.perf_counter()

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
