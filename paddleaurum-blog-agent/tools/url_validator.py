"""URL format validation and accessibility checking."""

import re
import requests
from urllib.parse import urlparse
from typing import Optional, Tuple

# Simple regex for URL format (does not guarantee existence)
URL_REGEX = re.compile(
    r"^(https?|ftp)://"  # scheme
    r"([a-zA-Z0-9.-]+)"   # domain
    r"(\.[a-zA-Z]{2,})?"  # tld
    r"(:\d+)?"            # port
    r"(/.*)?$",           # path
    re.IGNORECASE,
)


def validate_url(url: str) -> bool:
    """Check if the string is a valid URL format."""
    return bool(URL_REGEX.match(url))


def check_url_accessibility(
    url: str,
    timeout: float = 5.0,
    allow_redirects: bool = True,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Perform a HEAD request to verify the URL is accessible.

    Returns:
        (accessible, status_code, error_message)
    """
    if not validate_url(url):
        return False, None, "Invalid URL format"

    try:
        # Use HEAD request to be lightweight
        response = requests.head(
            url,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )
        # Consider 2xx and 3xx as accessible
        accessible = 200 <= response.status_code < 400
        return accessible, response.status_code, None
    except requests.exceptions.RequestException as e:
        return False, None, str(e)