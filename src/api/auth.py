"""
src/api/auth.py
──────────────────
Optional API-key check for mutating endpoints (POST /runs,
POST /customers/{id}/score) -- an X-API-Key header check, not a full auth
system (no accounts, no sessions, no roles). That's a deliberate scope
call: this is an internal tool with no user-facing login of its own today,
so a shared-secret header is the right-sized mechanism, not an
under-built one -- see CLAUDE.md's Infrastructure section for why
something heavier (OAuth, JWT, a users table) isn't done here.

Disabled by default (API_KEY unset) so local development and the
frontend, which does not send this header, keep working without extra
setup. Set API_KEY to require it -- every caller, including the
frontend's own requests, then needs X-API-Key: <value> or gets 401.
"""

import os
from typing import Optional

from fastapi import Header, HTTPException

API_KEY = os.environ.get("API_KEY")


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(401, "Missing or invalid X-API-Key header")
