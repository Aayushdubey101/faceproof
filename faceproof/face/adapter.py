"""Controlled interface to the face-recognition foundation in `face_engine/`.

Every call into `face_engine` goes through this module, so the rest of
FaceProof never depends on the foundation's own API surface. Imports are
deferred because loading the tensor backend is slow and `verify` never needs it.
"""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_MODEL = "ArcFace"
DEFAULT_DETECTOR = "retinaface"
DEFAULT_METRIC = "cosine"


def scan_face(
    image_path: str,
    model_name: str = DEFAULT_MODEL,
    detector_backend: str = DEFAULT_DETECTOR,
) -> Dict[str, Any]:
    """Detect, align and encode the face in `image_path`.

    Raises ValueError when the image contains no usable face.
    """
    from face_engine import DeepFace  # deferred: loading the tensor backend is slow

    embeddings = DeepFace.represent(
        img_path=image_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=True,
    )
    if not embeddings:
        raise ValueError(f"no face detected in {image_path}")

    primary = embeddings[0]
    return {
        "faces_detected": len(embeddings),
        "embedding_dimensions": len(primary["embedding"]),
        "facial_area": primary["facial_area"],
        "model": model_name,
        "detector": detector_backend,
    }


def compare_faces(
    probe_path: str,
    candidate_path: str,
    model_name: str = DEFAULT_MODEL,
    detector_backend: str = DEFAULT_DETECTOR,
    distance_metric: str = DEFAULT_METRIC,
) -> Dict[str, Any]:
    """Compare two faces. Returns {verified, distance, threshold}.

    Raises ValueError when the candidate image contains no detectable face -
    the caller decides whether that means "skip" or "fail".
    """
    from face_engine import DeepFace  # deferred: loading the tensor backend is slow

    result = DeepFace.verify(
        img1_path=probe_path,
        img2_path=candidate_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=True,
    )
    return {
        "verified": bool(result["verified"]),
        "distance": float(result["distance"]),
        "threshold": float(result["threshold"]),
    }
