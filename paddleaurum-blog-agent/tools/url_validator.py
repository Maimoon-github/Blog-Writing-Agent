# """tools/url_validator.py

# URL format validation and HTTP accessibility checking.

# `validate_url` performs a regex format check only — no network call.
# `check_url_accessibility` issues a HEAD request (falling back to GET if HEAD
# is refused or returns a 4xx/5xx) to determine whether a URL is reachable.
# This HEAD→GET fallback pattern mirrors the logic in nodes/citation_formatter.py
# so that the tool and the node always agree on URL validity.
# """

# from __future__ import annotations

# import re
# from typing import Optional, Tuple
# from urllib.parse import urlparse

# import requests

# # ── URL format regex ──────────────────────────────────────────────────────────
# _URL_PATTERN = re.compile(
#     r"^https?://"              # scheme — only http/https accepted
#     r"[a-zA-Z0-9.-]+"          # hostname (labels)
#     r"(\.[a-zA-Z]{2,})?"       # optional TLD
#     r"(:\d{1,5})?"             # optional port
#     r"(/[^\s]*)?$",            # optional path/query/fragment
#     re.IGNORECASE,
# )

# _DEFAULT_TIMEOUT: float = 8.0     # seconds — matches citation_formatter timeout
# _ACCESSIBLE_STATUSES: range = range(200, 400)  # 2xx and 3xx


# def validate_url(url: str) -> bool:
#     """
#     Return True if `url` matches a well-formed http/https URL pattern.

#     This is a pure format check — no network request is made.

#     Parameters
#     ----------
#     url : String to validate.

#     Returns
#     -------
#     bool
#     """
#     if not url or not isinstance(url, str):
#         return False
#     return bool(_URL_PATTERN.match(url.strip()))


# def check_url_accessibility(
#     url: str,
#     timeout: float = _DEFAULT_TIMEOUT,
# ) -> Tuple[bool, Optional[int], Optional[str]]:
#     """
#     Verify that `url` is reachable over HTTP/HTTPS.

#     Attempts a HEAD request first.  If the server refuses HEAD or returns
#     a non-successful status, falls back to a GET request — matching the
#     strategy used in nodes/citation_formatter.py.

#     Parameters
#     ----------
#     url     : The URL to probe.
#     timeout : Per-request timeout in seconds.

#     Returns
#     -------
#     (accessible: bool, status_code: int | None, error: str | None)
#         accessible  — True if the URL returned a 2xx or 3xx response.
#         status_code — Final HTTP status code, or None on connection failure.
#         error       — Human-readable error message, or None on success.
#     """
#     if not validate_url(url):
#         return False, None, "Invalid URL format."

#     # ── HEAD request ─────────────────────────────────────────────────────────
#     try:
#         head_resp = requests.head(url, timeout=timeout, allow_redirects=True)
#         if head_resp.status_code in _ACCESSIBLE_STATUSES:
#             return True, head_resp.status_code, None
#         # HEAD succeeded but returned an error status — fall through to GET
#     except requests.exceptions.RequestException:
#         pass  # server may not support HEAD — fall through to GET

#     # ── GET fallback ──────────────────────────────────────────────────────────
#     try:
#         get_resp = requests.get(
#             url,
#             timeout=timeout,
#             allow_redirects=True,
#             stream=True,   # avoid downloading the full body
#         )
#         get_resp.close()
#         accessible = get_resp.status_code in _ACCESSIBLE_STATUSES
#         return accessible, get_resp.status_code, None
#     except requests.exceptions.RequestException as exc:
#         return False, None, str(exc)
























# @##################################################################################################

















# tools/url_validator.py
"""tools/url_validator.py

URL format validation and HTTP accessibility checking.

`validate_url` performs a regex format check only — no network call.
`check_url_accessibility` issues a HEAD request (falling back to GET if HEAD
is refused or returns a 4xx/5xx) to determine whether a URL is reachable.
This HEAD→GET fallback pattern mirrors the logic in nodes/citation_formatter.py
so that the tool and the node always agree on URL validity.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

# ── URL format regex ──────────────────────────────────────────────────────────
_URL_PATTERN = re.compile(
    r"^https?://"              # scheme — only http/https accepted
    r"[a-zA-Z0-9.-]+"          # hostname (labels)
    r"(\.[a-zA-Z]{2,})?"       # optional TLD
    r"(:\d{1,5})?"             # optional port
    r"(/[^\s]*)?$",            # optional path/query/fragment
    re.IGNORECASE,
)

_DEFAULT_TIMEOUT: float = 8.0     # seconds — matches citation_formatter timeout
_ACCESSIBLE_STATUSES: range = range(200, 400)  # 2xx and 3xx


def validate_url(url: str) -> bool:
    """
    Return True if `url` matches a well-formed http/https URL pattern.

    This is a pure format check — no network request is made.

    Parameters
    ----------
    url : String to validate.

    Returns
    -------
    bool
    """
    if not url or not isinstance(url, str):
        return False
    return bool(_URL_PATTERN.match(url.strip()))


def check_url_accessibility(
    url: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Verify that `url` is reachable over HTTP/HTTPS.

    Attempts a HEAD request first.  If the server refuses HEAD or returns
    a non-successful status, falls back to a GET request — matching the
    strategy used in nodes/citation_formatter.py.

    Parameters
    ----------
    url     : The URL to probe.
    timeout : Per-request timeout in seconds.

    Returns
    -------
    (accessible: bool, status_code: int | None, error: str | None)
        accessible  — True if the URL returned a 2xx or 3xx response.
        status_code — Final HTTP status code, or None on connection failure.
        error       — Human-readable error message, or None on success.
    """
    if not validate_url(url):
        return False, None, "Invalid URL format."

    # ── HEAD request ─────────────────────────────────────────────────────────
    try:
        head_resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if head_resp.status_code in _ACCESSIBLE_STATUSES:
            return True, head_resp.status_code, None
        # HEAD succeeded but returned an error status — fall through to GET
    except requests.exceptions.RequestException:
        pass  # server may not support HEAD — fall through to GET

    # ── GET fallback ──────────────────────────────────────────────────────────
    try:
        get_resp = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            stream=True,   # avoid downloading the full body
        )
        get_resp.close()
        accessible = get_resp.status_code in _ACCESSIBLE_STATUSES
        return accessible, get_resp.status_code, None
    except requests.exceptions.RequestException as exc:
        return False, None, str(exc)