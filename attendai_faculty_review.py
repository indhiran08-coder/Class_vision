#!/usr/bin/env python3
"""AttendAI Faculty Review (local prototype).

Reviews existing AUTO_PRESENT rows without editing/deleting the source attendance
rows. Faculty decisions, roster decisions, and finalization events are stored in
separate SQLite tables. Run on localhost only; this prototype has no login system.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

APP_TITLE = "AttendAI | Faculty Attendance Review"
DB_DEFAULT = "data/attendai_attendance.db"
ROSTER_DEFAULT = "Student.csv"
REVIEW_OUTCOMES = [
    "PENDING",
    "VERIFIED_PRESENT",
    "CORRECTED_IDENTITY",
    "REJECTED_FALSE_POSITIVE",
]
ROSTER_OUTCOMES = ["MANUAL_PRESENT", "CONFIRMED_ABSENT"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def get_connection(path: str) -> sqlite3.Connection:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Attendance database not found: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_review_schema(conn: sqlite3.Connection) -> None:
    """Add separate faculty-review tables; do not alter core attendance rows."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS faculty_attendance_reviews (
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            attendance_id INTEGER NOT NULL REFERENCES attendance(attendance_id),
            original_student_reg_id TEXT NOT NULL,
            review_outcome TEXT NOT NULL CHECK(review_outcome IN
                ('VERIFIED_PRESENT','CORRECTED_IDENTITY','REJECTED_FALSE_POSITIVE')),
            corrected_student_reg_id TEXT,
            reviewer_name TEXT NOT NULL,
            reviewed_at_utc TEXT NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY(session_id, attendance_id),
            CHECK((review_outcome = 'CORRECTED_IDENTITY' AND corrected_student_reg_id IS NOT NULL)
               OR (review_outcome != 'CORRECTED_IDENTITY' AND corrected_student_reg_id IS NULL))
        );
        CREATE TABLE IF NOT EXISTS faculty_roster_decisions (
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            student_reg_id TEXT NOT NULL,
            roster_outcome TEXT NOT NULL CHECK(roster_outcome IN ('MANUAL_PRESENT','CONFIRMED_ABSENT')),
            reviewer_name TEXT NOT NULL,
            reviewed_at_utc TEXT NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY(session_id, student_reg_id)
        );
        CREATE TABLE IF NOT EXISTS faculty_session_finalizations (
            session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
            reviewer_name TEXT NOT NULL,
            finalized_at_utc TEXT NOT NULL,
            note TEXT NOT NULL,
            export_path TEXT NOT NULL,
            final_present_count INTEGER NOT NULL,
            final_absent_count INTEGER NOT NULL
        );
        """
    )
    conn.commit()


def load_roster(path: str) -> Tuple[Dict[str, dict], Optional[str]]:
    p = Path(path).expanduser()
    if not p.is_file():
        return {}, f"Roster file not found: {p.resolve()}"
    roster: Dict[str, dict] = {}
    try:
        with p.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "Reg_id" not in reader.fieldnames:
                return {}, "Roster CSV must contain a 'Reg_id' column."
            for row in reader:
                reg_id = (row.get("Reg_id") or "").strip()
                if not reg_id:
                    continue
                roster[reg_id] = {
                    "name": (row.get("name") or "").strip(),
                    "department": (row.get("department") or "").strip(),
                    "year": (row.get("year") or "").strip(),
                    "section": (row.get("section") or "").strip(),
                }
    except Exception as exc:
        return {}, f"Could not read roster CSV: {exc}"
    return roster, None


def log_audit(conn: sqlite3.Connection, session_id: str, event_type: str,
              reviewer: str, reg_id: Optional[str], details: dict,
              track_id: Optional[int] = None) -> None:
    conn.execute(
        """INSERT INTO audit_events
           (session_id, timestamp_utc, event_type, student_reg_id, person_track_id, details_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session_id, utc_now_iso(), event_type, reg_id, track_id,
         json.dumps({"reviewer": reviewer, **details}, sort_keys=True)),
    )


def fetch_sessions(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT s.session_id, s.session_name, s.started_at_utc, s.ended_at_utc, s.status,
                  COUNT(a.attendance_id) AS attendance_count
           FROM sessions s LEFT JOIN attendance a ON a.session_id = s.session_id
           GROUP BY s.session_id
           ORDER BY s.started_at_utc DESC"""
    ).fetchall()


def fetch_attendance(conn: sqlite3.Connection, session_id: str) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT a.*, r.review_outcome, r.corrected_student_reg_id,
                  r.reviewer_name, r.reviewed_at_utc, r.reason AS review_reason
           FROM attendance a LEFT JOIN faculty_attendance_reviews r
             ON r.session_id = a.session_id AND r.attendance_id = a.attendance_id
           WHERE a.session_id = ? ORDER BY a.marked_at_utc, a.attendance_id""",
        (session_id,),
    ).fetchall()


def fetch_roster_decisions(conn: sqlite3.Connection, session_id: str) -> Dict[str, sqlite3.Row]:
    rows = conn.execute(
        "SELECT * FROM faculty_roster_decisions WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    return {r["student_reg_id"]: r for r in rows}


def derive_roster_state(attendance_rows: List[sqlite3.Row],
                        roster: Dict[str, dict],
                        roster_decisions: Dict[str, sqlite3.Row]):
    """Compute a preview of final roster state without modifying source rows."""
    present_sources: Dict[str, List[str]] = {}
    unresolved_sources = set()
    pending_reviews = 0
    for row in attendance_rows:
        outcome = row["review_outcome"]
        original_id = row["student_reg_id"]
        if not outcome:
            pending_reviews += 1
            unresolved_sources.add(original_id)
        elif outcome == "VERIFIED_PRESENT":
            present_sources.setdefault(original_id, []).append("VERIFIED_PRESENT")
        elif outcome == "CORRECTED_IDENTITY":
            corrected_id = row["corrected_student_reg_id"]
            if corrected_id:
                present_sources.setdefault(corrected_id, []).append("CORRECTED_IDENTITY")
            unresolved_sources.add(original_id)
        elif outcome == "REJECTED_FALSE_POSITIVE":
            unresolved_sources.add(original_id)

    for reg_id, decision in roster_decisions.items():
        if decision["roster_outcome"] == "MANUAL_PRESENT":
            present_sources.setdefault(reg_id, []).append("MANUAL_PRESENT")

    all_ids = set(roster) | set(present_sources) | set(unresolved_sources) | set(roster_decisions)
    outcomes = {}
    for reg_id in sorted(all_ids):
        source = present_sources.get(reg_id, [])
        decision = roster_decisions.get(reg_id)
        if source and (len(source) > 1 or (decision and decision["roster_outcome"] == "CONFIRMED_ABSENT")):
            outcomes[reg_id] = ("CONFLICT", ",".join(source))
        elif source:
            outcomes[reg_id] = ("PRESENT", ",".join(source))
        elif decision and decision["roster_outcome"] == "CONFIRMED_ABSENT":
            outcomes[reg_id] = ("ABSENT", "CONFIRMED_ABSENT")
        else:
            outcomes[reg_id] = ("REVIEW_REQUIRED", "UNRESOLVED")
    duplicates = {rid: src for rid, src in present_sources.items() if len(src) > 1}
    return outcomes, pending_reviews, duplicates, present_sources



def verified_auto_present_ids(attendance_rows: List[sqlite3.Row]) -> set:
    """IDs supported by a faculty-approved automatic row, including corrected rows."""
    result = set()
    for row in attendance_rows:
        if row["review_outcome"] == "VERIFIED_PRESENT":
            result.add(row["student_reg_id"])
        elif row["review_outcome"] == "CORRECTED_IDENTITY" and row["corrected_student_reg_id"]:
            result.add(row["corrected_student_reg_id"])
    return result


def save_review(conn: sqlite3.Connection, session_id: str, row: sqlite3.Row,
                outcome: str, corrected_id: Optional[str], reviewer: str, reason: str) -> None:
    if outcome not in REVIEW_OUTCOMES[1:]:
        raise ValueError("Choose a completed review outcome.")
    if not reason.strip():
        raise ValueError("A reason is required for the audit trail.")
    if outcome == "CORRECTED_IDENTITY":
        if not corrected_id:
            raise ValueError("Select the corrected student ID.")
        if corrected_id == row["student_reg_id"]:
            raise ValueError("The corrected ID matches the original. Choose VERIFIED_PRESENT instead.")
    else:
        corrected_id = None
    now = utc_now_iso()
    conn.execute(
        """INSERT INTO faculty_attendance_reviews
           (session_id, attendance_id, original_student_reg_id, review_outcome,
            corrected_student_reg_id, reviewer_name, reviewed_at_utc, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(session_id, attendance_id) DO UPDATE SET
             original_student_reg_id=excluded.original_student_reg_id,
             review_outcome=excluded.review_outcome,
             corrected_student_reg_id=excluded.corrected_student_reg_id,
             reviewer_name=excluded.reviewer_name,
             reviewed_at_utc=excluded.reviewed_at_utc,
             reason=excluded.reason""",
        (session_id, row["attendance_id"], row["student_reg_id"], outcome,
         corrected_id, reviewer, now, reason.strip()),
    )
    log_audit(conn, session_id, "FACULTY_ATTENDANCE_REVIEW_SAVED", reviewer,
              row["student_reg_id"], {
                  "attendance_id": row["attendance_id"],
                  "review_outcome": outcome,
                  "corrected_student_reg_id": corrected_id,
                  "reason": reason.strip(),
              }, row["person_track_id"])
    conn.commit()


def save_roster_decision(conn: sqlite3.Connection, session_id: str, reg_id: str,
                         outcome: str, reviewer: str, reason: str) -> None:
    if outcome not in ROSTER_OUTCOMES:
        raise ValueError("Invalid roster decision.")
    if not reason.strip():
        raise ValueError("A reason is required for the audit trail.")
    now = utc_now_iso()
    conn.execute(
        """INSERT INTO faculty_roster_decisions
           (session_id, student_reg_id, roster_outcome, reviewer_name, reviewed_at_utc, reason)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(session_id, student_reg_id) DO UPDATE SET
             roster_outcome=excluded.roster_outcome,
             reviewer_name=excluded.reviewer_name,
             reviewed_at_utc=excluded.reviewed_at_utc,
             reason=excluded.reason""",
        (session_id, reg_id, outcome, reviewer, now, reason.strip()),
    )
    log_audit(conn, session_id, "FACULTY_ROSTER_DECISION_SAVED", reviewer, reg_id,
              {"roster_outcome": outcome, "reason": reason.strip()})
    conn.commit()



def clear_roster_decision(conn: sqlite3.Connection, session_id: str, reg_id: str,
                          reviewer: str) -> None:
    conn.execute(
        "DELETE FROM faculty_roster_decisions WHERE session_id = ? AND student_reg_id = ?",
        (session_id, reg_id),
    )
    log_audit(conn, session_id, "FACULTY_ROSTER_DECISION_CLEARED", reviewer, reg_id,
              {"reason": "Faculty cleared the prior manual roster decision."})
    conn.commit()


def make_final_csv(conn: sqlite3.Connection, session_id: str,
                   roster: Dict[str, dict], output_dir: str) -> Tuple[Path, int, int, int]:
    auto_rows = conn.execute(
        """SELECT a.*, r.review_outcome, r.corrected_student_reg_id,
                  r.reviewer_name AS row_reviewer, r.reviewed_at_utc, r.reason AS review_reason
           FROM attendance a LEFT JOIN faculty_attendance_reviews r
             ON r.session_id = a.session_id AND r.attendance_id = a.attendance_id
           WHERE a.session_id = ? ORDER BY a.marked_at_utc""",
        (session_id,),
    ).fetchall()
    roster_decisions = fetch_roster_decisions(conn, session_id)
    session = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    outcomes, pending_reviews, duplicates, sources = derive_roster_state(auto_rows, roster, roster_decisions)
    if pending_reviews:
        raise ValueError(f"{pending_reviews} automatic attendance record(s) still need faculty review.")
    if duplicates:
        raise ValueError("Conflicting present sources for: " + ", ".join(sorted(duplicates)))
    unresolved = [rid for rid, (status, _) in outcomes.items() if status in ("REVIEW_REQUIRED", "CONFLICT")]
    # Only registered roster members are required for a class register finalization.
    unresolved_roster = [rid for rid in roster if outcomes.get(rid, ("REVIEW_REQUIRED", ""))[0] in ("REVIEW_REQUIRED", "CONFLICT")]
    if unresolved_roster:
        raise ValueError(f"{len(unresolved_roster)} roster student(s) still need a present/absent decision.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"attendai_finalized_session_{session_id}.csv"
    fieldnames = [
        "session_id", "session_name", "session_status", "student_reg_id", "student_name",
        "department", "year", "section", "final_status", "decision_source",
        "attendance_marked_at_utc", "evidence_count", "evidence_share", "similarity",
        "face_result_age_seconds", "association_score", "faculty_review_outcome",
        "reviewer_name", "reviewed_at_utc", "review_reason",
    ]
    auto_by_reg: Dict[str, sqlite3.Row] = {}
    for row in auto_rows:
        outcome = row["review_outcome"]
        if outcome == "VERIFIED_PRESENT":
            auto_by_reg[row["student_reg_id"]] = row
        elif outcome == "CORRECTED_IDENTITY" and row["corrected_student_reg_id"]:
            auto_by_reg[row["corrected_student_reg_id"]] = row
    rows = []
    present_count = absent_count = 0
    for reg_id in sorted(roster):
        status, source = outcomes.get(reg_id, ("REVIEW_REQUIRED", "UNRESOLVED"))
        if status == "PRESENT":
            present_count += 1
        elif status == "ABSENT":
            absent_count += 1
        evidence = auto_by_reg.get(reg_id)
        manual = roster_decisions.get(reg_id)
        review_row = evidence
        rows.append({
            "session_id": session_id,
            "session_name": session["session_name"],
            "session_status": session["status"],
            "student_reg_id": reg_id,
            "student_name": roster[reg_id].get("name", ""),
            "department": roster[reg_id].get("department", ""),
            "year": roster[reg_id].get("year", ""),
            "section": roster[reg_id].get("section", ""),
            "final_status": status,
            "decision_source": source,
            "attendance_marked_at_utc": evidence["marked_at_utc"] if evidence else "",
            "evidence_count": evidence["evidence_count"] if evidence else "",
            "evidence_share": evidence["evidence_share"] if evidence else "",
            "similarity": evidence["similarity"] if evidence else "",
            "face_result_age_seconds": evidence["face_result_age_seconds"] if evidence else "",
            "association_score": evidence["association_score"] if evidence else "",
            "faculty_review_outcome": review_row["review_outcome"] if review_row else (manual["roster_outcome"] if manual else ""),
            "reviewer_name": review_row["row_reviewer"] if review_row else (manual["reviewer_name"] if manual else ""),
            "reviewed_at_utc": review_row["reviewed_at_utc"] if review_row else (manual["reviewed_at_utc"] if manual else ""),
            "review_reason": review_row["review_reason"] if review_row else (manual["reason"] if manual else ""),
        })
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path, present_count, absent_count, len(unresolved)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="✅", layout="wide")
    st.title("AttendAI — Faculty Review")
    st.caption("Review system attendance decisions before they become the faculty-approved class register.")
    st.warning(
        "Local prototype: no login/authentication is built in. Run on localhost only; "
        "do not expose this app to the campus network or internet without institutional authentication and role-based access."
    )

    with st.sidebar:
        st.header("Configuration")
        db_path = st.text_input("Attendance database", value=DB_DEFAULT)
        roster_path = st.text_input("Class roster CSV", value=ROSTER_DEFAULT)
        reviewer = st.text_input("Faculty reviewer name", value=st.session_state.get("reviewer_name", ""))
        st.session_state["reviewer_name"] = reviewer.strip()
        st.caption("Use the roster for the relevant class/section. The provided default roster is Student.csv.")

    if not reviewer.strip():
        st.info("Enter the faculty reviewer name in the sidebar to enable review actions.")

    roster, roster_error = load_roster(roster_path)
    if roster_error:
        st.warning(roster_error + " Roster correction and finalization require a valid roster.")

    try:
        conn = get_connection(db_path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    try:
        ensure_review_schema(conn)
        sessions = fetch_sessions(conn)
        if not sessions:
            st.info("No attendance sessions found in the selected database.")
            st.stop()

        labels = {
            f'{s["session_name"]} | {s["session_id"]} | {s["status"]} | {s["attendance_count"]} rows': s["session_id"]
            for s in sessions
        }
        chosen_label = st.selectbox("Select session", list(labels.keys()))
        session_id = labels[chosen_label]
        session = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        finalized = conn.execute("SELECT * FROM faculty_session_finalizations WHERE session_id = ?", (session_id,)).fetchone()
        if finalized:
            st.success(f"This session was finalized by {finalized['reviewer_name']} at {finalized['finalized_at_utc']}.")
            st.caption(f"Final report: {finalized['export_path']}")

        attendance_rows = fetch_attendance(conn, session_id)
        roster_decisions = fetch_roster_decisions(conn, session_id)
        auto_total = len(attendance_rows)
        pending_total = sum(1 for row in attendance_rows if not row["review_outcome"])
        verified_total = sum(1 for row in attendance_rows if row["review_outcome"] == "VERIFIED_PRESENT")
        corrected_total = sum(1 for row in attendance_rows if row["review_outcome"] == "CORRECTED_IDENTITY")
        rejected_total = sum(1 for row in attendance_rows if row["review_outcome"] == "REJECTED_FALSE_POSITIVE")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Auto-present records", auto_total)
        c2.metric("Awaiting review", pending_total)
        c3.metric("Verified", verified_total)
        c4.metric("Corrected", corrected_total)
        c5.metric("Rejected", rejected_total)

        tabs = st.tabs(["Review automatic records", "Roster reconciliation", "Audit log", "Finalize & export"])
        frozen = bool(finalized)
        with tabs[0]:
            st.subheader("Review each automatic attendance record")
            st.write("The original AUTO_PRESENT record is preserved. Your decision is stored separately with reviewer, time, and reason.")
            if not attendance_rows:
                st.info("This session has no automatic attendance rows.")
            for row in attendance_rows:
                reg_id = row["student_reg_id"]
                person = roster.get(reg_id, {})
                name = person.get("name") or "Name not found in roster"
                current = row["review_outcome"] or "PENDING"
                title = f'{reg_id} — {name} | {current} | similarity {row["similarity"]:.3f} | association {row["association_score"]:.3f}'
                with st.expander(title, expanded=(current == "PENDING")):
                    a, b, c = st.columns(3)
                    a.write(f'**Marked at (UTC):** {row["marked_at_utc"]}')
                    b.write(f'**Person track:** {row["person_track_id"]}')
                    c.write(f'**Evidence:** {row["evidence_count"]} observations; share {row["evidence_share"]:.2f}')
                    st.write(f'**Face result age:** {row["face_result_age_seconds"]:.3f}s')
                    st.write(f'**Original decision:** `{row["decision"]}`')
                    if row["reviewer_name"]:
                        st.info(f'Last review: {row["review_outcome"]} by {row["reviewer_name"]} at {row["reviewed_at_utc"]}. Reason: {row["review_reason"]}')
                    with st.form(f"review_{session_id}_{row['attendance_id']}"):
                        ix = REVIEW_OUTCOMES.index(current) if current in REVIEW_OUTCOMES else 0
                        chosen_outcome = st.selectbox("Faculty decision", REVIEW_OUTCOMES, index=ix, disabled=frozen)
                        roster_ids = sorted(set(roster) | {reg_id})
                        current_corrected = row["corrected_student_reg_id"] or reg_id
                        corrected_options = roster_ids or [reg_id]
                        corrected_index = corrected_options.index(current_corrected) if current_corrected in corrected_options else 0
                        corrected_id = st.selectbox("Correct identity to (used only for CORRECTED_IDENTITY)", corrected_options, index=corrected_index, disabled=frozen)
                        reason = st.text_area("Reason / evidence", value=row["review_reason"] or "", key=f"reason_{session_id}_{row['attendance_id']}", disabled=frozen, help="Required for every saved decision; for example, visually checked against the class register.")
                        submitted = st.form_submit_button("Save faculty decision", disabled=frozen or not reviewer.strip())
                    if submitted:
                        try:
                            save_review(conn, session_id, row, chosen_outcome, corrected_id, reviewer.strip(), reason)
                            st.success("Faculty decision saved and audit-logged.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

        with tabs[1]:
            st.subheader("Reconcile the selected class roster")
            if not roster:
                st.warning("Load a valid Student.csv roster before reconciling or finalizing.")
            else:
                outcomes, unresolved_reviews, dupes, sources = derive_roster_state(attendance_rows, roster, roster_decisions)
                rows_display = []
                for reg_id in sorted(roster):
                    status, source = outcomes.get(reg_id, ("REVIEW_REQUIRED", "UNRESOLVED"))
                    d = roster_decisions.get(reg_id)
                    rows_display.append({
                        "Reg ID": reg_id,
                        "Name": roster[reg_id].get("name", ""),
                        "Department": roster[reg_id].get("department", ""),
                        "Year": roster[reg_id].get("year", ""),
                        "Section": roster[reg_id].get("section", ""),
                        "Calculated status": status,
                        "Source": source,
                        "Manual decision": d["roster_outcome"] if d else "",
                    })
                st.dataframe(rows_display, use_container_width=True, hide_index=True)
                unresolved_ids = [r["Reg ID"] for r in rows_display if r["Calculated status"] in ("REVIEW_REQUIRED", "CONFLICT")]
                # Include students with prior manual decisions so faculty can change or clear them.
                editable_roster_ids = sorted(set(unresolved_ids) | set(roster_decisions))
                auto_present_ids = verified_auto_present_ids(attendance_rows)
                st.markdown("#### Record a faculty-confirmed roster decision")
                st.caption("Students not recognized automatically are not automatically considered absent. Select the student and explicitly confirm present or absent.")
                if editable_roster_ids:
                    with st.form(f"roster_decision_{session_id}"):
                        selected_id = st.selectbox("Student to resolve or edit", editable_roster_ids)
                        old_decision = roster_decisions.get(selected_id)
                        old_outcome = old_decision["roster_outcome"] if old_decision else "CONFIRMED_ABSENT"
                        default_idx = ROSTER_OUTCOMES.index(old_outcome) if old_outcome in ROSTER_OUTCOMES else 1
                        selected_outcome = st.selectbox("Decision", ROSTER_OUTCOMES, index=default_idx)
                        roster_reason = st.text_area("Reason / evidence", value=old_decision["reason"] if old_decision else "")
                        save_roster = st.form_submit_button("Save roster decision", disabled=frozen or not reviewer.strip())
                    if save_roster:
                        try:
                            if selected_outcome == "CONFIRMED_ABSENT" and selected_id in auto_present_ids:
                                raise ValueError("This student has an approved automatic-present source. Clear/review the conflict instead of marking absent.")
                            save_roster_decision(conn, session_id, selected_id, selected_outcome, reviewer.strip(), roster_reason)
                            st.success("Roster decision saved.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                    if selected_id in roster_decisions:
                        if st.button("Clear this student's manual roster decision", key=f"clear_roster_{session_id}_{selected_id}", disabled=frozen or not reviewer.strip()):
                            clear_roster_decision(conn, session_id, selected_id, reviewer.strip())
                            st.success("Manual roster decision cleared and audit-logged.")
                            st.rerun()
                    st.markdown("#### Batch-confirm remaining students absent")
                    st.caption("Use only after checking the class register or attendance evidence. No unrecorded student is automatically marked absent.")
                    with st.form(f"batch_absent_{session_id}"):
                        batch_ids = st.multiselect("Students you have checked and confirm absent", unresolved_ids)
                        batch_reason = st.text_input("Reason for this batch decision", value="Checked against faculty class register")
                        batch_save = st.form_submit_button("Save selected absences", disabled=frozen or not reviewer.strip() or not batch_ids)
                    if batch_save:
                        try:
                            for rid in batch_ids:
                                if rid in auto_present_ids:
                                    raise ValueError(f"{rid} already has a faculty-approved automatic-present source; resolve the conflict first.")
                            for rid in batch_ids:
                                save_roster_decision(conn, session_id, rid, "CONFIRMED_ABSENT", reviewer.strip(), batch_reason)
                            st.success(f"Saved {len(batch_ids)} confirmed-absent decisions.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                else:
                    st.success("Every roster student currently has a calculated present/absent outcome.")

        with tabs[2]:
            st.subheader("Session audit trail")
            events = conn.execute(
                """SELECT timestamp_utc, event_type, student_reg_id, person_track_id, details_json
                   FROM audit_events WHERE session_id = ? ORDER BY timestamp_utc DESC, event_id DESC LIMIT 500""",
                (session_id,),
            ).fetchall()
            st.dataframe([dict(x) for x in events], use_container_width=True, hide_index=True)

        with tabs[3]:
            st.subheader("Finalize the faculty-reviewed register")
            outcomes, pending_reviews, dupes, sources = derive_roster_state(attendance_rows, roster, roster_decisions)
            unresolved_roster = [rid for rid in roster if outcomes.get(rid, ("REVIEW_REQUIRED", ""))[0] in ("REVIEW_REQUIRED", "CONFLICT")]
            unknown_ids = sorted(set(sources) - set(roster))
            st.write(f"Automatic records still needing review: **{pending_reviews}**")
            st.write(f"Roster students requiring present/absent decision: **{len(unresolved_roster)}**")
            st.write(f"Duplicate/conflicting present sources: **{len(dupes)}**")
            if unknown_ids:
                st.warning("Student IDs outside this roster: " + ", ".join(unknown_ids))
            if session["status"] != "CLOSED":
                st.warning("Close the attendance session before finalization.")
            if finalized:
                st.success("This session is locked as finalized. To preserve the audit trail, changes are disabled in this prototype.")
                st.code(finalized["export_path"])
            else:
                can_finalize = bool(reviewer.strip() and roster and session["status"] == "CLOSED" and pending_reviews == 0 and not unresolved_roster and not dupes and not unknown_ids)
                note = st.text_area("Faculty finalization note", help="Optional note attached to the finalization audit event.")
                if st.button("Finalize session and export reviewed CSV", type="primary", disabled=not can_finalize):
                    try:
                        output_path, present_count, absent_count, _ = make_final_csv(
                            conn, session_id, roster, "reports"
                        )
                        now = utc_now_iso()
                        conn.execute(
                            """INSERT INTO faculty_session_finalizations
                               (session_id, reviewer_name, finalized_at_utc, note, export_path, final_present_count, final_absent_count)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (session_id, reviewer.strip(), now, note.strip(), str(output_path), present_count, absent_count),
                        )
                        log_audit(conn, session_id, "FACULTY_SESSION_FINALIZED", reviewer.strip(), None,
                                  {"note": note.strip(), "export_path": str(output_path),
                                   "final_present_count": present_count, "final_absent_count": absent_count})
                        conn.commit()
                        st.success(f"Session finalized. Present: {present_count}; absent: {absent_count}.")
                        st.code(str(output_path))
                        st.rerun()
                    except Exception as exc:
                        conn.rollback()
                        st.error(f"Could not finalize: {exc}")
                if not can_finalize:
                    st.info("Finalization becomes available when the session is closed, every automatic record is reviewed, every roster student has an explicit present/absent outcome, and there are no conflicting identities.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
