#!/usr/bin/env python3
"""AttendAI asynchronous person tracking + face recognition + session attendance.

Default mode is preview/dry-run and does not write attendance. To enable writes,
pass --enable-attendance and a --session-name (or resume an active --session-id).
YOLOv8 Nano + ByteTrack run on the main loop; SCRFD + ArcFace run in a worker.
Only fresh, geometrically consistent, repeatedly recognized identities may be marked.
"""

import argparse
import json
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from attendai.recognizer import FaceRecognizer


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return None
    return vector / norm


def recognize_face(face, recognizer):
    """Apply the current AttendAI recognition thresholds without changing them."""
    embedding = normalize(face.get("embedding"))
    bbox = np.asarray(face["bbox"], dtype=np.float32)
    confidence = float(face.get("confidence", 0.0))

    if embedding is None:
        return {
            "embedding": None, "bbox": bbox, "confidence": confidence,
            "status": "UNKNOWN", "reg_id": None, "candidate_id": None,
            "similarity": None, "reason": "invalid embedding",
        }

    scores = recognizer.embeddings @ embedding
    ranking = np.argsort(scores)[::-1]
    best = int(ranking[0])
    similarity = float(scores[best])
    margin = (
        similarity - float(scores[int(ranking[1])])
        if len(ranking) > 1 else float("inf")
    )
    candidate_id = str(recognizer.reg_ids[best])

    if confidence < recognizer.min_detection_confidence:
        status, reason = "REVIEW", "low detection confidence"
    elif similarity < recognizer.min_similarity:
        status, reason = "UNKNOWN", "similarity below threshold"
    elif margin < recognizer.min_similarity_margin:
        status, reason = "REVIEW", "small recognition margin"
    else:
        status, reason = "MATCH", "initial thresholds passed"

    return {
        "embedding": embedding, "bbox": bbox, "confidence": confidence,
        "status": status, "reg_id": candidate_id if status == "MATCH" else None,
        "candidate_id": candidate_id, "similarity": similarity,
        "margin": margin, "reason": reason,
    }


@dataclass
class PersonMemory:
    """Recognition history and explicit state for a temporary ByteTrack ID."""
    track_id: int
    last_seen: float
    recent_matches: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    stable_reg_id: Optional[str] = None
    identity_conflict: bool = False
    state: str = "UNPROCESSED"

    def observe_face(self, detection):
        status = str(detection.get("status", "")).upper()
        reg_id = detection.get("reg_id")

        if status == "UNKNOWN":
            if self.stable_reg_id is None:
                self.state = "UNKNOWN"
            return

        if status == "REVIEW":
            self.state = "REVIEW_REQUIRED"
            return

        if status != "MATCH" or not reg_id:
            return

        self.recent_matches.append(reg_id)
        counts = Counter(self.recent_matches)
        winner, votes = counts.most_common(1)[0]
        share = votes / len(self.recent_matches)

        if self.stable_reg_id is None:
            if votes >= 3 and share >= 0.70:
                self.stable_reg_id = winner
                self.state = "KNOWN"
            else:
                self.state = "UNPROCESSED"
            return

        # Detect competing identity evidence independently of the majority winner.
        competing_votes = sum(
            count for identity, count in counts.items()
            if identity != self.stable_reg_id
        )
        competing_share = competing_votes / len(self.recent_matches)

        if competing_votes >= 2 and competing_share >= 0.40:
            self.identity_conflict = True
            self.state = "REVIEW_REQUIRED"
            return

        stable_votes = counts.get(self.stable_reg_id, 0)
        stable_share = stable_votes / len(self.recent_matches)

        if stable_votes >= 3 and stable_share >= 0.60:
            self.identity_conflict = False
            self.state = "KNOWN"
        elif self.identity_conflict:
            self.state = "REVIEW_REQUIRED"
        else:
            self.state = "UNPROCESSED"


def associate_face_to_person(face_bbox, people):
    """Associate a face with one plausible person box using geometry.

    Returns (person_track_id, score, person_bbox). Ambiguous associations are
    rejected. Coordinates are in the original camera frame.
    """
    fx1, fy1, fx2, fy2 = [float(v) for v in face_bbox]
    face_cx = (fx1 + fx2) / 2.0
    face_cy = (fy1 + fy2) / 2.0
    candidates = []

    for person in people:
        if person["track_id"] is None:
            continue
        px1, py1, px2, py2 = [float(v) for v in person["bbox"]]
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)
        person_cx = (px1 + px2) / 2.0
        y_relative = (face_cy - py1) / ph
        if face_cx < px1 - 0.08 * pw or face_cx > px2 + 0.08 * pw:
            continue
        if y_relative < -0.05 or y_relative > 0.62:
            continue
        x_score = max(0.0, 1.0 - abs(face_cx - person_cx) / (0.60 * pw))
        y_score = max(0.0, 1.0 - abs(y_relative - 0.23) / 0.39)
        score = 0.65 * x_score + 0.35 * y_score
        candidates.append((score, int(person["track_id"]), np.asarray(person["bbox"]).copy()))

    if not candidates:
        return None, 0.0, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_id, best_bbox = candidates[0]
    if len(candidates) > 1 and (best_score - candidates[1][0]) < 0.12:
        return None, best_score, None
    if best_score < 0.38:
        return None, best_score, None
    return best_id, best_score, best_bbox


def utc_now_iso():
    """UTC timestamp for persisted attendance/audit rows."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def box_center_and_size(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1))


def current_person_box_is_consistent(sample_box, current_box, max_shift_factor=0.45):
    """Guard against attaching an old face result to a body that moved/switched.

    This is deliberately conservative: it requires a plausible center movement
    and compatible box scale. It cannot prove physical identity continuity.
    """
    if sample_box is None or current_box is None:
        return False, float("inf")
    sx, sy, sw, sh = box_center_and_size(sample_box)
    cx, cy, cw, ch = box_center_and_size(current_box)
    shift = ((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5
    max_shift = max(100.0, max_shift_factor * max(sh, ch))
    width_ratio = cw / sw
    height_ratio = ch / sh
    scale_ok = 0.45 <= width_ratio <= 2.2 and 0.45 <= height_ratio <= 2.2
    return shift <= max_shift and scale_ok, shift


class AttendanceStore:
    """Local SQLite attendance store with session uniqueness and audit history.

    No face images or embeddings are stored here; only student IDs, evidence
    metadata, timestamps and decisions are retained.
    """
    def __init__(self, db_path, session_name=None, session_id=None):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._event_last_logged = {}
        self._create_schema()

        if session_id:
            row = self.conn.execute(
                "SELECT session_id, session_name, status FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                self.conn.close()
                raise ValueError("Session ID not found in database: " + str(session_id))
            if row["status"] != "ACTIVE":
                self.conn.close()
                raise ValueError("Session is not active and cannot be resumed: " + str(session_id))
            self.session_id = row["session_id"]
            self.session_name = row["session_name"]
        else:
            if not session_name:
                self.conn.close()
                raise ValueError("A session name is required when creating a session")
            self.session_id = uuid.uuid4().hex[:12]
            self.session_name = session_name
            self.conn.execute(
                "INSERT INTO sessions(session_id, session_name, started_at_utc, status) VALUES (?, ?, ?, 'ACTIVE')",
                (self.session_id, self.session_name, utc_now_iso()),
            )
            self.conn.commit()

    def _create_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                ended_at_utc TEXT,
                status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'CLOSED'))
            );
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                student_reg_id TEXT NOT NULL,
                marked_at_utc TEXT NOT NULL,
                person_track_id INTEGER NOT NULL,
                evidence_count INTEGER NOT NULL,
                evidence_share REAL NOT NULL,
                similarity REAL NOT NULL,
                face_result_age_seconds REAL NOT NULL,
                association_score REAL NOT NULL,
                decision TEXT NOT NULL DEFAULT 'AUTO_PRESENT',
                UNIQUE(session_id, student_reg_id)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                timestamp_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                student_reg_id TEXT,
                person_track_id INTEGER,
                details_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);
            CREATE INDEX IF NOT EXISTS idx_audit_session_time ON audit_events(session_id, timestamp_utc);
        """)
        self.conn.commit()

    def log_event(self, event_type, reg_id=None, track_id=None, details=None,
                  throttle_seconds=5.0, dedupe_key=None):
        """Persist a concise audit event; throttle repeating events without losing face hypotheses."""
        key = (event_type, dedupe_key if dedupe_key is not None else (reg_id or ""),
               track_id if track_id is not None else -1)
        now_mono = time.monotonic()
        last = self._event_last_logged.get(key)
        if last is not None and now_mono - last < throttle_seconds:
            return
        self._event_last_logged[key] = now_mono
        self.conn.execute(
            "INSERT INTO audit_events(session_id, timestamp_utc, event_type, student_reg_id, person_track_id, details_json) VALUES (?, ?, ?, ?, ?, ?)",
            (self.session_id, utc_now_iso(), event_type, reg_id, track_id,
             json.dumps(details or {}, sort_keys=True, ensure_ascii=True)),
        )
        self.conn.commit()

    def log_face_review(self, face, track_id, link, result_age, force_ambiguous=False):
        """Queue an unknown/uncertain face for faculty triage; never writes attendance."""
        status = str(face.get("status", "REVIEW"))
        event_type = "UNKNOWN_FACE_REVIEW" if status == "UNKNOWN" and not force_ambiguous else "AMBIGUOUS_FACE_REVIEW"
        bbox = face.get("bbox")
        cell = "unlocated"
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cell = f"{int(((x1 + x2) / 2) // 100)}:{int(((y1 + y2) / 2) // 100)}"
        except Exception:
            pass
        candidate_id = face.get("candidate_id")
        details = {
            "recognition_status": status,
            "reason": str(face.get("reason") or "Recognition or association requires review"),
            "candidate_id": str(candidate_id) if candidate_id else None,
            "similarity": round(float(face["similarity"]), 4) if face.get("similarity") is not None else None,
            "margin": round(float(face["margin"]), 4) if face.get("margin") is not None else None,
            "face_detection_confidence": round(float(face.get("confidence", 0.0)), 4),
            "geometry_score": round(float(link.get("geometry_score") or 0.0), 4),
            "face_result_age_seconds": round(float(result_age), 3),
        }
        # student_reg_id remains NULL: candidate_id is only an unverified suggestion.
        dedupe = f"{status}:{candidate_id or 'none'}:{details['reason']}:{cell}"
        self.log_event(event_type, reg_id=None, track_id=track_id, details=details,
                       throttle_seconds=5.0, dedupe_key=dedupe)

    def has_attendance(self, reg_id):
        row = self.conn.execute(
            "SELECT 1 FROM attendance WHERE session_id = ? AND student_reg_id = ? LIMIT 1",
            (self.session_id, reg_id),
        ).fetchone()
        return row is not None

    def record_present(self, reg_id, track_id, evidence_count, evidence_share,
                       similarity, result_age, association_score):
        """Insert one present row per registered student per session."""
        now_utc = utc_now_iso()
        try:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO attendance(
                    session_id, student_reg_id, marked_at_utc, person_track_id,
                    evidence_count, evidence_share, similarity,
                    face_result_age_seconds, association_score, decision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'AUTO_PRESENT')""",
                (self.session_id, reg_id, now_utc, int(track_id), int(evidence_count),
                 float(evidence_share), float(similarity), float(result_age), float(association_score)),
            )
            if cur.rowcount == 1:
                self.conn.execute(
                    "INSERT INTO audit_events(session_id, timestamp_utc, event_type, student_reg_id, person_track_id, details_json) VALUES (?, ?, 'AUTO_PRESENT_MARKED', ?, ?, ?)",
                    (self.session_id, now_utc, reg_id, int(track_id), json.dumps({
                        "evidence_count": int(evidence_count),
                        "evidence_share": round(float(evidence_share), 4),
                        "similarity": round(float(similarity), 4),
                        "face_result_age_seconds": round(float(result_age), 3),
                        "association_score": round(float(association_score), 4),
                    }, sort_keys=True)),
                )
                self.conn.commit()
                return "MARKED_PRESENT"
            self.conn.commit()
            return "ALREADY_PRESENT"
        except Exception:
            self.conn.rollback()
            raise

    def attendance_count(self):
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM attendance WHERE session_id = ?",
            (self.session_id,),
        ).fetchone()
        return int(row["n"])

    def close_session(self):
        if self.conn is None:
            return
        try:
            self.conn.execute(
                "UPDATE sessions SET status = 'CLOSED', ended_at_utc = ? WHERE session_id = ? AND status = 'ACTIVE'",
                (utc_now_iso(), self.session_id),
            )
            self.conn.commit()
        finally:
            self.conn.close()
            self.conn = None



class FaceWorker(threading.Thread):
    """One background SCRFD/ArcFace worker with bounded, drop-stale input."""
    def __init__(self, detector, recognizer, face_scale):
        super().__init__(name="AttendAI-FaceWorker", daemon=True)
        self.detector = detector
        self.recognizer = recognizer
        self.face_scale = face_scale
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.busy = threading.Event()

    def submit(self, frame, frame_number, capture_time, people):
        # Keep the worker from building a backlog of old camera frames.
        if self.busy.is_set() or not self.jobs.empty():
            return False
        job = {
            "frame": frame.copy(),
            "frame_number": frame_number,
            "capture_time": capture_time,
            "people": [
                {"bbox": p["bbox"].copy(), "track_id": p["track_id"],
                 "confidence": p["confidence"]}
                for p in people
            ],
        }
        try:
            self.jobs.put_nowait(job)
            return True
        except queue.Full:
            return False

    def run(self):
        while not self.stop_event.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            self.busy.set()
            started = time.monotonic()
            try:
                frame = job["frame"]
                height, width = frame.shape[:2]
                if self.face_scale != 1.0:
                    detect_frame = cv2.resize(
                        frame,
                        (max(1, int(width * self.face_scale)),
                         max(1, int(height * self.face_scale))),
                    )
                else:
                    detect_frame = frame

                raw_faces = self.detector.extract_faces(detect_frame)
                faces = [recognize_face(raw, self.recognizer) for raw in raw_faces]
                if self.face_scale != 1.0:
                    inv_scale = 1.0 / self.face_scale
                    for face in faces:
                        face["bbox"] = face["bbox"] * inv_scale

                links = []
                for face in faces:
                    person_id, geometry_score, person_bbox = associate_face_to_person(
                        face["bbox"], job["people"]
                    )
                    links.append({
                        "person_id": person_id,
                        "geometry_score": geometry_score,
                        "person_bbox": person_bbox,
                    })

                result = {
                    "frame_number": job["frame_number"],
                    "capture_time": job["capture_time"],
                    "finished_time": time.monotonic(),
                    "duration_ms": (time.monotonic() - started) * 1000.0,
                    "faces": faces,
                    "links": links,
                    "error": None,
                }
            except Exception as exc:  # surface worker errors in the UI/terminal
                result = {
                    "frame_number": job["frame_number"],
                    "capture_time": job["capture_time"],
                    "finished_time": time.monotonic(),
                    "duration_ms": (time.monotonic() - started) * 1000.0,
                    "faces": [], "links": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                self.busy.clear()
                self.jobs.task_done()

            try:
                self.results.put_nowait(result)
            except queue.Full:
                try:
                    self.results.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.results.put_nowait(result)
                except queue.Full:
                    pass

    def stop(self):
        self.stop_event.set()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Asynchronous AttendAI person tracking + face recognition preview (no attendance writes)."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--person-model", default="yolov8n.pt", help="YOLO person model path")
    parser.add_argument("--person-imgsz", type=int, default=256, help="YOLO inference size")
    parser.add_argument("--person-conf", type=float, default=0.40, help="YOLO person confidence")
    parser.add_argument("--face-scale", type=float, default=0.60,
                        help="Scale used for face detection (0.1-1.0)")
    parser.add_argument("--face-every", type=int, default=5,
                        help="Submit a face sample every N displayed camera frames")
    parser.add_argument("--person-retain-seconds", type=float, default=8.0,
                        help="Keep identity memory this long after a person track disappears")
    parser.add_argument("--enable-attendance", action="store_true",
                        help="Enable local SQLite attendance writes (off by default)")
    parser.add_argument("--session-name", default=None,
                        help="Human-readable class/session name when creating a session")
    parser.add_argument("--session-id", default=None,
                        help="Resume an existing ACTIVE session ID from the database")
    parser.add_argument("--attendance-db", default="data/attendai_attendance.db",
                        help="SQLite path for attendance and audit data")
    parser.add_argument("--attendance-max-result-age", type=float, default=1.5,
                        help="Maximum age in seconds for a face result to affect attendance")
    parser.add_argument("--attendance-max-box-shift-factor", type=float, default=0.45,
                        help="Allowed sampled-to-current person-box center shift as a fraction of box height")
    parser.add_argument("--attendance-min-votes", type=int, default=3,
                        help="Minimum consistent accepted face matches (minimum allowed: 3)")
    parser.add_argument("--attendance-min-share", type=float, default=0.70,
                        help="Minimum share of recent accepted matches for an identity")
    args = parser.parse_args()
    if args.enable_attendance and not (args.session_name or args.session_id):
        parser.error("--enable-attendance requires --session-name or --session-id")
    if args.session_id and not args.enable_attendance:
        parser.error("--session-id requires --enable-attendance")
    if args.attendance_max_result_age <= 0:
        parser.error("--attendance-max-result-age must be positive")
    if not 0.1 <= args.attendance_max_box_shift_factor <= 1.5:
        parser.error("--attendance-max-box-shift-factor must be between 0.1 and 1.5")
    if args.attendance_min_votes < 3:
        parser.error("--attendance-min-votes must be at least 3")
    if not 0.5 <= args.attendance_min_share <= 1.0:
        parser.error("--attendance-min-share must be between 0.5 and 1.0")
    return args


def main():
    args = parse_args()
    if args.person_imgsz < 160:
        raise SystemExit("--person-imgsz should be at least 160")
    if not 0.1 <= args.face_scale <= 1.0:
        raise SystemExit("--face-scale must be between 0.1 and 1.0")
    if args.face_every < 1:
        raise SystemExit("--face-every must be 1 or greater")
    if args.person_retain_seconds <= 0:
        raise SystemExit("--person-retain-seconds must be positive")

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA unavailable; YOLO will use CPU.")
    else:
        print("CUDA device:", torch.cuda.get_device_name(0))

    print("Loading YOLO person model:", args.person_model)
    person_model = YOLO(args.person_model)
    print("Loading AttendAI SCRFD + ArcFace recognizer...")
    recognizer = FaceRecognizer()
    worker = FaceWorker(recognizer.detector, recognizer, args.face_scale)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        raise SystemExit("Could not open camera. Check /dev/video0, permissions, or another app using it.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    memories: Dict[int, PersonMemory] = {}
    stable_labels_by_reg: Dict[str, int] = {}
    next_stable_label = 1

    cached_faces = []
    cached_links = []
    last_face_capture_time = 0.0
    last_face_duration_ms = 0.0
    last_face_error = None
    attendance_store = None
    if args.enable_attendance:
        try:
            attendance_store = AttendanceStore(
                args.attendance_db,
                session_name=args.session_name,
                session_id=args.session_id,
            )
        except Exception as exc:
            worker.stop()
            cap.release()
            raise SystemExit(f"Could not initialize attendance session: {exc}")
        print("ATTENDANCE WRITES ENABLED")
        print("Session name:", attendance_store.session_name)
        print("Session ID:", attendance_store.session_id)
        print("Attendance DB:", Path(args.attendance_db).resolve())
    else:
        print("ATTENDANCE DRY RUN: no attendance database writes will occur.")
        print("To record attendance, add --enable-attendance --session-name 'Class / Period'.")
    preview_ready_ids = set()

    frames = 0
    fps = 0.0
    fps_window_start = time.monotonic()
    fps_window_frames = 0
    print("Starting asynchronous face worker and live preview.")
    print("YOLO + ByteTrack run on the display loop; face inference runs in a worker.")
    print("No video is saved. Press q to quit.")
    if attendance_store is None:
        print("DRY RUN: attendance candidates may be shown in the terminal; database writes are disabled.")
    else:
        print("ATTENDANCE ENABLED for this session; records and audit events are written to SQLite.")
    worker.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            loop_started = time.monotonic()
            capture_time = time.monotonic()
            height, width = frame.shape[:2]

            person_result = person_model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                device=device,
                classes=[0],
                imgsz=args.person_imgsz,
                conf=args.person_conf,
                verbose=False,
            )[0]

            people = []
            boxes = person_result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                confidences = boxes.conf.detach().cpu().tolist()
                for box, track_id, conf in zip(xyxy, track_ids, confidences):
                    track_id = int(track_id) if track_id is not None else None
                    people.append({
                        "bbox": box.astype(np.float32),
                        "track_id": track_id,
                        "confidence": float(conf),
                    })
                    if track_id is not None:
                        if track_id not in memories:
                            memories[track_id] = PersonMemory(track_id=track_id, last_seen=capture_time)
                        memories[track_id].last_seen = capture_time

            current_people_by_id = {
                p["track_id"]: p for p in people if p["track_id"] is not None
            }

            # Submit only people who still need recognition or attendance evidence.
            # Keep recognizing a known student until attendance is recorded.
            # Revalidate conflicting identities instead of silently skipping them.
            people_to_recognize = []
            for person in people:
                track_id = person["track_id"]
                if track_id is None:
                    people_to_recognize.append(person)
                    continue

                memory = memories.get(track_id)
                if memory is None or memory.stable_reg_id is None:
                    people_to_recognize.append(person)
                    continue

                if memory.identity_conflict:
                    people_to_recognize.append(person)
                    continue

                already_present = (
                    attendance_store is not None
                    and attendance_store.has_attendance(memory.stable_reg_id)
                )

                if not already_present:
                    people_to_recognize.append(person)

            # Associate faces against ALL current person detections.
            # Attendance eligibility is checked later by the existing
            # stable-identity, conflict, and attendance safeguards.
            # The bounded worker queue prevents stale frames accumulating.
            if people and frames % args.face_every == 0:
                worker.submit(frame, frames, capture_time, people)

            # Apply the most recent completed face result on the main thread.
            try:
                result = worker.results.get_nowait()
            except queue.Empty:
                result = None

            if result is not None:
                cached_faces = result["faces"]
                cached_links = result["links"]
                last_face_capture_time = result["capture_time"]
                last_face_duration_ms = result["duration_ms"]
                last_face_error = result["error"]
                result_age = max(0.0, time.monotonic() - result["capture_time"])

                if last_face_error:
                    print("Face worker error:", last_face_error)
                elif result_age > args.attendance_max_result_age:
                    if attendance_store is not None:
                        attendance_store.log_event(
                            "STALE_FACE_RESULT_IGNORED",
                            details={"age_seconds": round(result_age, 3),
                                     "limit_seconds": args.attendance_max_result_age},
                        )
                else:
                    # Result age alone is not sufficient: the same person-track
                    # must still exist and its body box must remain geometrically
                    # consistent with the sampled box used to link the face.
                    for face, link in zip(cached_faces, cached_links):
                        person_id = link["person_id"]
                        if person_id is None:
                            if attendance_store is not None:
                                attendance_store.log_face_review(
                                    face, None, link, result_age, force_ambiguous=True
                                )
                                if face["status"] == "MATCH":
                                    attendance_store.log_event(
                                        "FACE_NOT_ASSOCIATED_TO_PERSON",
                                        reg_id=face.get("reg_id"),
                                        details={"result_age_seconds": round(result_age, 3),
                                                 "geometry_score": link.get("geometry_score", 0.0)},
                                    )
                            continue

                        # Reject a result if its ByteTrack ID is not in the current frame.
                        current_person = current_people_by_id.get(person_id)
                        memory = memories.get(person_id)
                        if current_person is None or memory is None:
                            if memory is not None:
                                memory.state = "REVIEW_REQUIRED"
                            if attendance_store is not None:
                                attendance_store.log_face_review(
                                    face, person_id, link, result_age, force_ambiguous=True
                                )
                                if face["status"] == "MATCH":
                                    attendance_store.log_event(
                                        "PERSON_TRACK_NOT_CURRENT",
                                        reg_id=face.get("reg_id"), track_id=person_id,
                                        details={"result_age_seconds": round(result_age, 3)},
                                    )
                            continue

                        geometry_score = float(link.get("geometry_score") or 0.0)
                        source_person_box = link.get("person_bbox")
                        consistent_box, center_shift = current_person_box_is_consistent(
                            source_person_box,
                            current_person["bbox"],
                            args.attendance_max_box_shift_factor,
                        )
                        if source_person_box is None or geometry_score < 0.38 or not consistent_box:
                            memory.state = "REVIEW_REQUIRED"
                            if attendance_store is not None:
                                attendance_store.log_face_review(
                                    face, person_id, link, result_age, force_ambiguous=True
                                )
                                if face["status"] == "MATCH":
                                    attendance_store.log_event(
                                        "PERSON_BOX_CONTINUITY_REVIEW",
                                        reg_id=face.get("reg_id"), track_id=person_id,
                                        details={"result_age_seconds": round(result_age, 3),
                                                 "geometry_score": round(geometry_score, 4),
                                                 "center_shift_pixels": round(center_shift, 1)},
                                    )
                            continue

                        memory.observe_face(face)

                        # Save fresh UNKNOWN/REVIEW results for faculty triage. They
                        # remain separate from attendance and cannot mark anyone present.
                        if face["status"] != "MATCH" or not face.get("reg_id"):
                            if attendance_store is not None:
                                attendance_store.log_face_review(
                                    face, person_id, link, result_age
                                )
                            continue

                        if memory.stable_reg_id and memory.stable_reg_id not in stable_labels_by_reg:
                            stable_labels_by_reg[memory.stable_reg_id] = next_stable_label
                            next_stable_label += 1

                        # Only accepted detections matching the repeated stable identity
                        # are candidates; UNKNOWN/REVIEW results never mark attendance.
                        if face["status"] != "MATCH" or not face.get("reg_id"):
                            continue
                        if memory.stable_reg_id != face["reg_id"]:
                            if memory.stable_reg_id is not None:
                                memory.state = "REVIEW_REQUIRED"
                            if attendance_store is not None:
                                attendance_store.log_event(
                                    "IDENTITY_NOT_STABLE_OR_MISMATCH",
                                    reg_id=face.get("reg_id"), track_id=person_id,
                                    details={"stable_reg_id": memory.stable_reg_id,
                                             "result_age_seconds": round(result_age, 3)},
                                )
                            continue
                        if memory.identity_conflict:
                            if attendance_store is not None:
                                attendance_store.log_event(
                                    "IDENTITY_CONFLICT_REVIEW",
                                    reg_id=memory.stable_reg_id, track_id=person_id,
                                    details={"result_age_seconds": round(result_age, 3)},
                                )
                            continue

                        votes = Counter(memory.recent_matches)
                        evidence_count = votes.get(memory.stable_reg_id, 0)
                        evidence_share = evidence_count / max(1, len(memory.recent_matches))
                        if evidence_count < args.attendance_min_votes or evidence_share < args.attendance_min_share:
                            if attendance_store is not None:
                                attendance_store.log_event(
                                    "INSUFFICIENT_REPEAT_EVIDENCE",
                                    reg_id=memory.stable_reg_id, track_id=person_id,
                                    details={"evidence_count": evidence_count,
                                             "evidence_share": round(evidence_share, 4)},
                                )
                            continue

                        # If two current person tracks claim the same student ID,
                        # do not choose between them automatically.
                        competing_tracks = [
                            tid for tid, other in memories.items()
                            if tid != person_id
                            and other.stable_reg_id == memory.stable_reg_id
                            and not other.identity_conflict
                            and (time.monotonic() - other.last_seen) <= 1.0
                        ]
                        if competing_tracks:
                            if attendance_store is not None:
                                attendance_store.log_event(
                                    "DUPLICATE_ACTIVE_IDENTITY_CLAIM_REVIEW",
                                    reg_id=memory.stable_reg_id, track_id=person_id,
                                    details={"competing_track_ids": competing_tracks},
                                )
                            continue

                        similarity = face.get("similarity")
                        if similarity is None:
                            continue
                        candidate_details = {
                            "session_id": attendance_store.session_id if attendance_store else None,
                            "student_reg_id": memory.stable_reg_id,
                            "person_track_id": person_id,
                            "evidence_count": evidence_count,
                            "evidence_share": round(evidence_share, 4),
                            "similarity": round(float(similarity), 4),
                            "face_result_age_seconds": round(result_age, 3),
                            "association_score": round(geometry_score, 4),
                        }
                        if attendance_store is None:
                            if memory.stable_reg_id not in preview_ready_ids:
                                print("[DRY RUN] Attendance candidate only; no write:",
                                      json.dumps(candidate_details, sort_keys=True))
                                preview_ready_ids.add(memory.stable_reg_id)
                        else:
                            outcome = attendance_store.record_present(
                                reg_id=memory.stable_reg_id,
                                track_id=person_id,
                                evidence_count=evidence_count,
                                evidence_share=evidence_share,
                                similarity=float(similarity),
                                result_age=result_age,
                                association_score=geometry_score,
                            )
                            if outcome == "MARKED_PRESENT":
                                print("[ATTENDANCE] Marked present:", memory.stable_reg_id,
                                      "session=", attendance_store.session_id,
                                      "votes=", evidence_count,
                                      "age=", round(result_age, 2), "s")
                            # ALREADY_PRESENT is intentionally silent to avoid log spam.

            now = time.monotonic()
            stale = [tid for tid, memory in memories.items()
                     if now - memory.last_seen > args.person_retain_seconds]
            for tid in stale:
                del memories[tid]

            annotated = frame.copy()
            for person in people:
                x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
                track_id = person["track_id"]
                memory = memories.get(track_id) if track_id is not None else None
                color = (0, 165, 255) if memory is not None and memory.identity_conflict else (0, 200, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                if track_id is None:
                    label = "Person (ByteTrack ID pending)"
                elif memory is not None and memory.stable_reg_id:
                    logical_id = stable_labels_by_reg.setdefault(memory.stable_reg_id, next_stable_label)
                    if logical_id == next_stable_label:
                        next_stable_label += 1
                    if memory.identity_conflict:
                        label = f"T{logical_id} | ID CONFLICT - REVIEW | BT P{track_id}"
                    else:
                        label = f"T{logical_id} | ID {memory.stable_reg_id} | BT P{track_id}"
                else:
                    label = f"BT P{track_id} | identity pending"
                cv2.putText(annotated, label, (max(0, x1), max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

            # Draw face boxes only if the source sample is recent enough; otherwise
            # rely on the current body box so an old face box doesn't visibly lag.
            face_age = (now - last_face_capture_time) if last_face_capture_time else None
            if face_age is not None and face_age <= 0.35 and not last_face_error:
                for face, link in zip(cached_faces, cached_links):
                    person_id = link["person_id"]
                    source_person_box = link["person_bbox"]
                    current_person = current_people_by_id.get(person_id)
                    if source_person_box is not None and current_person is not None:
                        old_cx = (source_person_box[0] + source_person_box[2]) / 2.0
                        old_cy = (source_person_box[1] + source_person_box[3]) / 2.0
                        current_box = current_person["bbox"]
                        new_cx = (current_box[0] + current_box[2]) / 2.0
                        new_cy = (current_box[1] + current_box[3]) / 2.0
                        dx, dy = new_cx - old_cx, new_cy - old_cy
                        adjusted_box = face["bbox"] + np.array([dx, dy, dx, dy], dtype=np.float32)
                    else:
                        adjusted_box = face["bbox"]
                    x1, y1, x2, y2 = [int(v) for v in adjusted_box]
                    status = face["status"]
                    color = (0, 200, 0) if status == "MATCH" else ((0, 165, 255) if status == "UNKNOWN" else (0, 0, 255))
                    candidate = face["reg_id"] or face["candidate_id"] or "UNKNOWN"
                    sim = face["similarity"]
                    sim_text = f"{sim:.2f}" if sim is not None else "n/a"
                    if person_id is not None and person_id in memories and memories[person_id].stable_reg_id:
                        logical_id = stable_labels_by_reg.get(memories[person_id].stable_reg_id, "?")
                        link_text = f"T{logical_id}"
                    elif person_id is not None:
                        link_text = f"P{person_id}"
                    else:
                        link_text = "unlinked"
                    label = f"{link_text} | {status} | {candidate} | {sim_text}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (max(0, x1), max(18, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

            loop_ms = (time.monotonic() - loop_started) * 1000.0
            face_count = len(cached_faces)
            cv2.rectangle(annotated, (8, 8), (690, 111), (0, 0, 0), -1)
            cv2.putText(annotated, f"AttendAI LIVE | people: {len(people)} | last face sample: {face_count}",
                        (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"Main loop: {loop_ms:.0f} ms | Wall FPS: {fps:.1f} | async SCRFD/ArcFace",
                        (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
            age_text = f"Face result age: {face_age:.2f}s | Face inference: {last_face_duration_ms:.0f} ms" if face_age is not None else "Face worker warming up..."
            cv2.putText(annotated, age_text, (16, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if attendance_store is None:
                mode_text = "ATTENDANCE DRY RUN - no database writes"
            else:
                mode_text = f"SESSION {attendance_store.session_id} | marked present: {attendance_store.attendance_count()}"
            cv2.putText(annotated, mode_text, (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (0, 220, 255) if attendance_store is None else (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("AttendAI - Async Person Tracking + Face Recognition", annotated)
            key = cv2.waitKey(1) & 0xFF

            frames += 1
            fps_window_frames += 1
            wall_now = time.monotonic()
            wall_elapsed = wall_now - fps_window_start
            if wall_elapsed >= 1.0:
                fps = fps_window_frames / wall_elapsed
                fps_window_frames = 0
                fps_window_start = wall_now
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\nStopping AttendAI preview...")
    finally:
        worker.stop()
        worker.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. Displayed frames:", frames)
        if attendance_store is not None:
            marked = attendance_store.attendance_count()
            print("Attendance session:", attendance_store.session_id)
            print("Marked present in this session:", marked)
            attendance_store.close_session()
            print("Session closed. Database:", Path(args.attendance_db).resolve())
        else:
            print("Dry run finished. No attendance rows were written.")


if __name__ == "__main__":
    main()
