"""HTTP transfer of images: probe upload out, candidate images in."""

from __future__ import annotations

import os
from typing import Optional

import requests

USER_AGENT = "FaceProof/0.1 (+https://github.com/)"
REQUEST_TIMEOUT = 60
MAX_IMAGE_BYTES = 20 * 1024 * 1024

UPLOAD_ENDPOINT = "https://0x0.st"
UPLOAD_FALLBACK_ENDPOINT = "https://catbox.moe/user/api.php"


def upload_probe(image_path: str) -> str:
    """Publish the probe image temporarily and return its public URL.

    Reverse image search providers fetch the probe over HTTP, so a local file is
    not enough.

    ponytail: two free anonymous hosts, no API key. Swap in object storage with
    signed URLs if either starts rate-limiting.
    """
    headers = {"User-Agent": USER_AGENT}

    try:
        with open(image_path, "rb") as handle:
            response = requests.post(
                UPLOAD_ENDPOINT,
                files={"file": handle},
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        url = response.text.strip()
        if response.ok and url.startswith("http"):
            return url
    except requests.RequestException:
        pass  # fall through to the backup host

    try:
        with open(image_path, "rb") as handle:
            response = requests.post(
                UPLOAD_FALLBACK_ENDPOINT,
                data={"reqtype": "fileupload"},
                files={"fileToUpload": handle},
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
    except requests.RequestException:
        raise RuntimeError("probe upload failed: both anonymous hosts unreachable") from None

    url = response.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"probe upload failed: {url[:200]!r}")
    return url


def download_image(url: str, dest_dir: str, name: str) -> Optional[str]:
    """Fetch a candidate image to disk. Returns None on any transport failure.

    An unavailable or oversized candidate is skipped, never fatal.
    """
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, stream=True
        )
        response.raise_for_status()
        payload = response.raw.read(MAX_IMAGE_BYTES + 1, decode_content=True)
    except (requests.RequestException, OSError):
        return None

    if not payload or len(payload) > MAX_IMAGE_BYTES:
        return None

    path = os.path.join(dest_dir, name)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path
