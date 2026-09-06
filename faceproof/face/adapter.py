"""Controlled interface to the face-recognition foundation in `face_engine/`.

Every call into `face_engine` goes through this module, so the rest of
FaceProof never depends on the foundation's own API surface. Imports are
deferred because loading the tensor backend is slow and `verify` never needs it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union, cast

DEFAULT_MODEL = "ArcFace"
DEFAULT_DETECTOR = "retinaface"
DEFAULT_METRIC = "cosine"


def scan_face(
    image_path: str,
    model_name: str = DEFAULT_MODEL,
    detector_backend: str = DEFAULT_DETECTOR,
) -> Dict[str, Any]:
    """Detect, align and encode the face in `image_path`.

    The returned `embedding` is the probe vector itself, so a caller can compare
    against it without paying for detection and encoding again. It is None when
    the image holds several faces: the foundation then compares every probe face
    against every candidate face, and one vector cannot stand in for that.

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

    # represent() is typed as either shape; a single image always yields the flat one
    primary = cast(Dict[str, Any], embeddings[0])
    single_face = len(embeddings) == 1
    return {
        "faces_detected": len(embeddings),
        "embedding": [float(value) for value in primary["embedding"]] if single_face else None,
        "embedding_dimensions": len(primary["embedding"]),
        "facial_area": primary["facial_area"],
        "model": model_name,
        "detector": detector_backend,
    }


def compare_faces(
    probe: Union[str, Sequence[float]],
    candidate_path: str,
    model_name: str = DEFAULT_MODEL,
    detector_backend: str = DEFAULT_DETECTOR,
    distance_metric: str = DEFAULT_METRIC,
) -> Dict[str, Any]:
    """Compare two faces. Returns {verified, distance, threshold}.

    `probe` is either an image path or an embedding from `scan_face`. The
    foundation scores both forms through the same distance and threshold, so a
    pre-encoded probe only skips work - it does not change the verdict.

    Raises ValueError when the candidate image contains no detectable face -
    the caller decides whether that means "skip" or "fail".
    """
    from face_engine import DeepFace  # deferred: loading the tensor backend is slow

    probe_input: Union[str, List[float]] = probe if isinstance(probe, str) else list(probe)
    result = DeepFace.verify(
        img1_path=probe_input,
        img2_path=candidate_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=True,
        silent=True,  # a pre-encoded probe is deliberate here, not worth a warning
    )
    return {
        "verified": bool(result["verified"]),
        "distance": float(result["distance"]),
        "threshold": float(result["threshold"]),
    }


def verification_threshold(
    model_name: str = DEFAULT_MODEL,
    distance_metric: str = DEFAULT_METRIC,
) -> float:
    """Retrieve the authoritative threshold for a given model and distance metric.

    Queries the face-recognition foundation directly so that FaceProof never
    maintains a duplicate or desynchronized threshold configuration.
    """
    from face_engine.modules.verification import find_threshold

    return float(find_threshold(model_name, distance_metric))

