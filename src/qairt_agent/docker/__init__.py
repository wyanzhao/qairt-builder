"""Docker wrapper for the QAIRT agent.

Provides the pinned worker-image contract, the runtime mount plan, and an
injectable ``docker run`` wrapper.  The framework fails closed when Docker is
unavailable, never installs Docker, and mounts (rather than bakes) the SDK,
models, and artifacts.
"""

from qairt_agent.docker.image import (
    DEFAULT_IMAGE_REF,
    DEFAULT_PLATFORM,
    MODELS_MOUNT_TARGET,
    PINNED_PYTHON,
    PINNED_UBUNTU,
    BindMount,
    DockerImageConfig,
    RuntimeMounts,
    WORKER_PYTHONPATH,
    WorkerImageConfig,
    build_context_excludes,
    default_mounts,
    image_provenance,
    resolve_image_digest,
)
from qairt_agent.docker.runner import DockerRunner

__all__ = [
    "DEFAULT_IMAGE_REF",
    "DEFAULT_PLATFORM",
    "BindMount",
    "DockerImageConfig",
    "DockerRunner",
    "WorkerImageConfig",
    "MODELS_MOUNT_TARGET",
    "PINNED_PYTHON",
    "PINNED_UBUNTU",
    "RuntimeMounts",
    "WORKER_PYTHONPATH",
    "build_context_excludes",
    "default_mounts",
    "image_provenance",
    "resolve_image_digest",
]
