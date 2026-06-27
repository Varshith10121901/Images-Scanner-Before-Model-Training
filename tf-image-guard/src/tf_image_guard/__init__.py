"""
tf-image-guard
==============
A fault-tolerant CLI toolkit that validates and filters corrupted images
for TensorFlow pipelines.

Features:
- 3-layer validation: filesystem → Pillow → TensorFlow
- Never halts execution — all errors are caught and logged
- Parallel scanning with configurable workers
- Quarantine system with manifest generation
- Reports in console, JSON, and CSV formats
"""

__version__ = "0.1.0"
__author__ = "tf-image-guard contributors"
__license__ = "MIT"

# Lazy imports — heavy deps (TF, Pillow) are only loaded when actually used
__all__ = [
    "__version__",
    "cli",
    "scanner",
    "validators",
    "quarantine",
    "reporter",
    "utils",
]
