"""
Per-application (thread) context for document analysis so the agent keeps track
of which company and which documents belong to the same application.
Uses thread_id (e.g. application_id) as key. In-memory store with optional TTL.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
import threading

# thread_id -> { "application_company_name": str, "documents": [ {"document_type": str, "company_match": bool|None, "companies_mentioned": str} ], "updated_at": datetime }
_store: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
# Keep context for 2 hours after last update
_TTL_HOURS = 2


def _ttl_cleanup():
    with _lock:
        now = datetime.now(timezone.utc)
        to_del = [
            tid for tid, data in _store.items()
            if (now - data.get("updated_at", now)).total_seconds() > _TTL_HOURS * 3600
        ]
        for tid in to_del:
            del _store[tid]


def get_thread_context(thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not thread_id:
        return None
    _ttl_cleanup()
    with _lock:
        return _store.get(thread_id)


def update_thread_context(
    thread_id: str,
    application_company_name: Optional[str],
    document_type: str,
    company_match: Optional[bool] = None,
    companies_mentioned: Optional[str] = None,
) -> None:
    with _lock:
        if thread_id not in _store:
            _store[thread_id] = {
                "application_company_name": application_company_name,
                "documents": [],
                "updated_at": datetime.now(timezone.utc),
            }
        _store[thread_id]["documents"].append({
            "document_type": document_type,
            "company_match": company_match,
            "companies_mentioned": companies_mentioned or "",
        })
        _store[thread_id]["updated_at"] = datetime.now(timezone.utc)
        if application_company_name:
            _store[thread_id]["application_company_name"] = application_company_name


def build_previous_documents_prompt(thread_context: Dict[str, Any]) -> str:
    docs = thread_context.get("documents") or []
    if not docs:
        return ""
    app_company = thread_context.get("application_company_name") or "Unknown"
    lines = [
        f"This application is for the company: \"{app_company}\".",
        "You have already analyzed the following documents in this application:",
    ]
    for i, d in enumerate(docs, 1):
        match_str = "COMPANY_MATCH" if d.get("company_match") else ("COMPANY_MISMATCH" if d.get("company_match") is False else "unknown")
        companies = d.get("companies_mentioned") or "—"
        lines.append(f"- Document {i}: {d.get('document_type', 'unknown')} — {match_str}; companies mentioned: {companies}")
    lines.append(
        "Ensure the current document is for the SAME application company. If it refers to a different company, output COMPANY_MISMATCH."
    )
    return "\n".join(lines)
