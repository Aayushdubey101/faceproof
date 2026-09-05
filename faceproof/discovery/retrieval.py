"""HTTP transfer of images: probe upload out, candidate images in.

Every call here is bounded by a timeout, and the probe is always uploaded as an
optimized copy - a phone photo is several megabytes, which is slow enough on a
free host to time out mid-demo.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests
from PIL import Image, ImageOps

from faceproof.discovery import DiscoveryError

USER_AGENT = "FaceProof/0.1 (+https://github.com/)"

# (connect, read) seconds - no external call may hang the demo indefinitely
SEARCH_TIMEOUT = (10, 60)
UPLOAD_TIMEOUT = (10, 90)
DOWNLOAD_TIMEOUT = (5, 20)

UPLOAD_ATTEMPTS = 2  # one retry, for a transient hiccup only
RETRY_DELAY_SECONDS = 1.5
MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Probe optimization. 1600px keeps the face far larger than ArcFace's 112x112
# input and stays well inside what Google Lens indexes; quality 88 is visually
# clean while cutting a typical 2 MB phone photo to a few hundred KB.
MAX_PROBE_PIXELS = 1600
PROBE_JPEG_QUALITY = 88
OPTIMIZED_PROBE_NAME = "probe.jpg"

PROVIDER_ENV = "FACE_UPLOAD_PROVIDER"
DEFAULT_PROVIDER = "catbox"

# Each provider is one multipart POST whose response body is the public URL.
# A table, not a plugin system: adding a host is three literals.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    # permanent anonymous host
    "catbox": {
        "endpoint": "https://catbox.moe/user/api.php",
        "field": "fileToUpload",
        "data": {"reqtype": "fileupload"},
    },
    # same operator, file expires after an hour - preferable for a probe image
    "litterbox": {
        "endpoint": "https://litterbox.catbox.moe/resources/internals/api.php",
        "field": "fileToUpload",
        "data": {"reqtype": "fileupload", "time": "1h"},
    },
    # kept selectable, but it answers "uploads disabled" (HTTP 503) as of 2026-09
    "0x0": {
        "endpoint": "https://0x0.st",
        "field": "file",
        "data": {},
    },
}

TIMEOUT_NOTE = (
    f"{UPLOAD_TIMEOUT[0]}s connect / {UPLOAD_TIMEOUT[1]}s read, {UPLOAD_ATTEMPTS} attempts"
)


def selected_provider() -> str:
    """The upload provider this run will use (FACE_UPLOAD_PROVIDER, else the default)."""
    return (os.environ.get(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()


def optimize_probe(image_path: str, dest_dir: str) -> str:
    """Write an upload-sized JPEG copy of the probe into `dest_dir`.

    The original file is never touched - the pipeline hashes it, scans it and
    matches against it, so only what leaves the machine for reverse search is
    resized. EXIF is dropped, with orientation baked into the pixels first so
    the published face is not rotated away from the one that was scanned.
    """
    with Image.open(image_path) as opened:
        upright = ImageOps.exif_transpose(opened) or opened
        image = upright.convert("RGB")
        image.thumbnail((MAX_PROBE_PIXELS, MAX_PROBE_PIXELS), Image.Resampling.LANCZOS)
        path = os.path.join(dest_dir, OPTIMIZED_PROBE_NAME)
        image.save(path, format="JPEG", quality=PROBE_JPEG_QUALITY, optimize=True)
    return path


def upload_probe(image_path: str, provider: Optional[str] = None) -> str:
    """Publish the probe image temporarily and return its public URL.

    Reverse image search providers fetch the probe over HTTP, so a local file is
    not enough. Exactly one provider is attempted - a host that is known to be
    disabled must not silently cost the demo a timeout.

    ponytail: free anonymous hosts, no API key and no availability guarantee.
    Swap in object storage with signed URLs if uptime has to be underwritten.
    """
    name = (provider or selected_provider()).strip().lower()
    config = PROVIDERS.get(name)
    if config is None:
        raise DiscoveryError(
            name, f"unknown upload provider - set {PROVIDER_ENV} to one of: {', '.join(PROVIDERS)}"
        )

    reason = "upload was never attempted"
    for attempt in range(UPLOAD_ATTEMPTS):
        if attempt:
            time.sleep(RETRY_DELAY_SECONDS)
        try:
            with open(image_path, "rb") as handle:
                response = requests.post(
                    config["endpoint"],
                    data=config["data"],
                    files={config["field"]: (os.path.basename(image_path), handle, "image/jpeg")},
                    headers={"User-Agent": USER_AGENT},
                    timeout=UPLOAD_TIMEOUT,
                )
        except requests.Timeout:
            reason = f"upload timed out after {UPLOAD_TIMEOUT[1]}s"
            continue
        except requests.RequestException:
            # never str(error): only the type is guaranteed free of request detail
            reason = f"could not reach {config['endpoint']}"
            continue

        body = response.text.strip()
        if response.ok and body.startswith("http"):
            return body
        reason = f"HTTP {response.status_code} - {body[:120] or 'empty response'}"

    raise DiscoveryError(name, reason, TIMEOUT_NOTE)


def download_image(url: str, dest_dir: str, name: str) -> Optional[str]:
    """Fetch a candidate image to disk. Returns None on any transport failure.

    An unavailable, oversized or slow candidate is skipped, never fatal: one bad
    thumbnail must not end the investigation.
    """
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=DOWNLOAD_TIMEOUT, stream=True
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
