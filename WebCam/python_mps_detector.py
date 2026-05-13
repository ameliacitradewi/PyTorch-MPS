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
    if 0 in names_map and "person" in names_map[0].lower():
        target_ids.append(0)
    return sorted(set(target_ids))


def emit_json(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def clamp_box_xyxy(x1, y1, x2, y2, width, height):
    cx1 = max(0.0, min(float(x1), float(width)))
    cy1 = max(0.0, min(float(y1), float(height)))
    cx2 = max(0.0, min(float(x2), float(width)))
    cy2 = max(0.0, min(float(y2), float(height)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def build_tracker_dets(boxes, width, height, np_module):
    rows = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            confidence = float(box.conf.item()) if box.conf is not None else 0.0
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            clamped = clamp_box_xyxy(x1, y1, x2, y2, width, height)
            if clamped is None:
                continue
            cx1, cy1, cx2, cy2 = clamped
            rows.append([cx1, cy1, cx2, cy2, confidence, float(cls_id)])
    if rows:
        return np_module.asarray(rows, dtype=np_module.float32)
    return np_module.empty((0, 6), dtype=np_module.float32)


def extract_track_features(tracker, np_module):
    features = {}
    for track in getattr(tracker, "active_tracks", []):
        bot_track_id = int(getattr(track, "id", getattr(track, "track_id", -1)))
        if bot_track_id < 0:
            continue
        smooth_feat = getattr(track, "smooth_feat", None)
        if smooth_feat is None:
            continue
        feat = np_module.asarray(smooth_feat, dtype=np_module.float32)
        if feat.size == 0:
            continue
        norm = float(np_module.linalg.norm(feat))
        if norm <= 1e-12:
            continue
        features[bot_track_id] = feat / norm
    return features


def load_reid_memory(memory_path, np_module):
    if not memory_path.exists():
        return 1, {}
    try:
        payload = json.loads(memory_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pylint: disable=broad-except
        log_stderr(f"[ReIDMemory] Failed to read {memory_path}: {type(exc).__name__}: {exc}")
        return 1, {}

    if not isinstance(payload, dict):
        return 1, {}

    next_id = int(payload.get("next_id", 1))
    if next_id < 1:
        next_id = 1

    profiles = {}
    for item in payload.get("profiles", []):
        if not isinstance(item, dict):
            continue
        stable_id = int(item.get("id", -1))
        if stable_id < 1:
            continue
        emb_raw = item.get("embedding")
        if not isinstance(emb_raw, list) or not emb_raw:
            continue
        emb = np_module.asarray(emb_raw, dtype=np_module.float32)
        norm = float(np_module.linalg.norm(emb))
        if norm <= 1e-12:
            continue
        emb = emb / norm
        profiles[stable_id] = {
            "embedding": emb,
            "seen_count": int(item.get("seen_count", 0)),
            "last_seen_frame": int(item.get("last_seen_frame", -1)),
        }
        if stable_id >= next_id:
            next_id = stable_id + 1
    return next_id, profiles


def save_reid_memory(memory_path, next_id, profiles):
    payload = {"next_id": int(next_id), "profiles": []}
    for stable_id, profile in sorted(profiles.items()):
        embedding = profile.get("embedding")
        if embedding is None:
            continue
        payload["profiles"].append(
            {
                "id": int(stable_id),
                "embedding": embedding.tolist(),
                "seen_count": int(profile.get("seen_count", 0)),
                "last_seen_frame": int(profile.get("last_seen_frame", -1)),
            }
        )
    try:
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = memory_path.with_suffix(memory_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload), encoding="utf-8")
        temp_path.replace(memory_path)
    except Exception as exc:  # pylint: disable=broad-except
        log_stderr(f"[ReIDMemory] Failed to write {memory_path}: {type(exc).__name__}: {exc}")


class StableIdentityStore:
    def __init__(
        self,
        np_module,
        profiles,
        next_id,
        match_threshold,
        ema,
        max_profiles,
        mapping_ttl_frames=600,
    ):
        self.np = np_module
        self.profiles = profiles
        self.next_id = next_id
        self.match_threshold = match_threshold
        self.ema = ema
        self.max_profiles = max_profiles
        self.mapping_ttl_frames = mapping_ttl_frames
        self.track_to_stable = {}
        self.track_last_seen = {}
        self.dirty = False

    def mark_seen(self, bot_track_id, frame_seq):
        self.track_last_seen[bot_track_id] = frame_seq

    def _best_profile_match(self, feature, used_stable_ids):
        best_id = None
        best_score = -1.0
        for candidate_id, candidate in self.profiles.items():
            if candidate_id in used_stable_ids:
                continue
            emb = candidate.get("embedding")
            if emb is None:
                continue
            score = float(self.np.dot(feature, emb))
            if score > best_score:
                best_score = score
                best_id = candidate_id
        return best_id, best_score

    def resolve(self, bot_track_id, feature, frame_seq, used_stable_ids):
        stable_id = self.track_to_stable.get(bot_track_id)
        if stable_id is None or stable_id not in self.profiles:
            best_id = None
            best_score = -1.0
            if feature is not None:
                best_id, best_score = self._best_profile_match(feature, used_stable_ids)
            if best_id is not None and best_score >= self.match_threshold:
                stable_id = best_id
            else:
                stable_id = self.next_id
                self.next_id += 1
                self.profiles[stable_id] = {
                    "embedding": feature.copy() if feature is not None else None,
                    "seen_count": 0,
                    "last_seen_frame": frame_seq,
                }
                self.dirty = True
        self.track_to_stable[bot_track_id] = stable_id
        return stable_id

    def update_profile(self, stable_id, feature, frame_seq):
        profile = self.profiles.setdefault(
            stable_id,
            {"embedding": None, "seen_count": 0, "last_seen_frame": frame_seq},
        )
        profile["seen_count"] = int(profile.get("seen_count", 0)) + 1
        profile["last_seen_frame"] = frame_seq
        if feature is None:
            return
        prev = profile.get("embedding")
        if prev is None:
            profile["embedding"] = feature.copy()
            self.dirty = True
            return
        blended = ((1.0 - self.ema) * prev) + (self.ema * feature)
        norm = float(self.np.linalg.norm(blended))
        if norm > 1e-12:
            profile["embedding"] = blended / norm
            self.dirty = True

    def cleanup_stale_mappings(self, frame_seq):
        stale_track_ids = [
            track_id
            for track_id, last_seen in self.track_last_seen.items()
            if (frame_seq - last_seen) > self.mapping_ttl_frames
        ]
        for track_id in stale_track_ids:
            self.track_last_seen.pop(track_id, None)
            self.track_to_stable.pop(track_id, None)

    def prune_profiles(self, protected_ids):
        if len(self.profiles) <= self.max_profiles:
            return
        overflow = len(self.profiles) - self.max_profiles
        candidates = sorted(
            self.profiles.items(),
            key=lambda item: int(item[1].get("last_seen_frame", -1)),
        )
        for stable_id, _ in candidates:
            if overflow <= 0:
                break
            if stable_id in protected_ids:
                continue
            self.profiles.pop(stable_id, None)
            overflow -= 1
            self.dirty = True

    def maybe_save(self, memory_path, frame_seq, save_interval):
        if self.dirty and (frame_seq % save_interval == 0):
            save_reid_memory(memory_path, self.next_id, self.profiles)
            self.dirty = False

    def flush(self, memory_path):
        if self.dirty:
            save_reid_memory(memory_path, self.next_id, self.profiles)
            self.dirty = False


def resolve_reid_paths(script_dir, yolo_config_dir, reid_model_path, reid_memory_path_env, path_module):
    reid_weights = path_module(reid_model_path).expanduser() if reid_model_path else path_module("osnet_x0_25_msmt17.pt")
    if not reid_weights.is_absolute():
        reid_weights = script_dir / reid_weights
    if not reid_weights.exists():
        fallback = script_dir / "osnet_x0_25_msmt17.pt"
        if fallback.exists():
            reid_weights = fallback
        elif yolo_config_dir:
            reid_weights = path_module(yolo_config_dir) / "osnet_x0_25_msmt17.pt"
    reid_weights = reid_weights.resolve(strict=False)

    if reid_memory_path_env:
        reid_memory_path = path_module(reid_memory_path_env).expanduser()
        if not reid_memory_path.is_absolute():
            reid_memory_path = script_dir / reid_memory_path
    elif yolo_config_dir:
        reid_memory_path = path_module(yolo_config_dir) / "reid_identity_memory.json"
    else:
        reid_memory_path = script_dir / "reid_identity_memory.json"
    reid_memory_path = reid_memory_path.resolve(strict=False)
    return reid_weights, reid_memory_path


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
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    try:
        from boxmot import BotSORT  # pylint: disable=import-outside-toplevel
    except ImportError:
        from boxmot import BotSort as BotSORT  # pylint: disable=import-outside-toplevel

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
    reid_model_path = os.environ.get("REID_MODEL_PATH", "osnet_x0_25_msmt17.pt").strip()
    reid_memory_path_env = os.environ.get("REID_MEMORY_PATH", "").strip()
    reid_match_threshold = float(os.environ.get("REID_MEMORY_MATCH_THRESHOLD", "0.7"))
    reid_memory_ema = float(os.environ.get("REID_MEMORY_EMA", "0.2"))
    reid_memory_max_profiles = int(os.environ.get("REID_MEMORY_MAX_PROFILES", "512"))
    reid_save_interval = max(1, int(os.environ.get("REID_MEMORY_SAVE_INTERVAL_FRAMES", "30")))

    script_dir = Path(__file__).resolve().parent
    reid_weights, reid_memory_path = resolve_reid_paths(
        script_dir=script_dir,
        yolo_config_dir=yolo_config_dir,
        reid_model_path=reid_model_path,
        reid_memory_path_env=reid_memory_path_env,
        path_module=Path,
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
    tracker = BotSORT(
        reid_weights=reid_weights,
        device=device,
        half=False,
        per_class=False,
    )
    log_stderr(f"[ReID] Loaded: {reid_weights.name} on {device}")

    next_id, profiles = load_reid_memory(reid_memory_path, np)
    identity_store = StableIdentityStore(
        np_module=np,
        profiles=profiles,
        next_id=next_id,
        match_threshold=reid_match_threshold,
        ema=reid_memory_ema,
        max_profiles=reid_memory_max_profiles,
    )
    log_stderr(f"[ReIDMemory] Loaded {len(identity_store.profiles)} profiles from {reid_memory_path}")

    emit_json(
        {
            "ready": True,
            "device": device,
            "mps_available": bool(torch.backends.mps.is_available()),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "reid_model": reid_weights.name,
        }
    )

    perf_window_count = 0
    perf_decode_total_ms = 0.0
    perf_infer_total_ms = 0.0
    perf_post_total_ms = 0.0
    perf_total_total_ms = 0.0
    perf_log_interval = 20
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
            frame_seq += 1

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

            dets_array = build_tracker_dets(result.boxes, width, height, np)
            tracks = tracker.update(dets_array, frame)
            if tracks is None:
                tracks = np.empty((0, 7), dtype=np.float32)

            features = extract_track_features(tracker, np)
            detections = []
            used_stable_ids = set()
            for track_row in tracks:
                if len(track_row) < 7:
                    continue
                clamped = clamp_box_xyxy(track_row[0], track_row[1], track_row[2], track_row[3], width, height)
                if clamped is None:
                    continue
                x1, y1, x2, y2 = clamped

                bot_track_id = int(track_row[4])
                confidence = float(track_row[5])
                cls_id = int(track_row[6])
                label = names_map.get(cls_id, f"class_{cls_id}")
                feature = features.get(bot_track_id)

                identity_store.mark_seen(bot_track_id, frame_seq)
                stable_id = identity_store.resolve(bot_track_id, feature, frame_seq, used_stable_ids)
                used_stable_ids.add(stable_id)
                identity_store.update_profile(stable_id, feature, frame_seq)

                detections.append(
                    {
                        "label": label,
                        "track_id": int(stable_id),
                        "confidence": confidence,
                        "x": x1 / width,
                        "y": y1 / height,
                        "w": (x2 - x1) / width,
                        "h": (y2 - y1) / height,
                    }
                )

            identity_store.cleanup_stale_mappings(frame_seq)
            identity_store.prune_profiles(used_stable_ids)
            identity_store.maybe_save(reid_memory_path, frame_seq, reid_save_interval)
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

    identity_store.flush(reid_memory_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_error:  # pylint: disable=broad-except
        log_stderr("Fatal detector error: " + str(fatal_error))
        log_stderr(traceback.format_exc())
        emit_json({"frame_id": -1, "detections": [], "error": str(fatal_error)})
        sys.exit(1)
