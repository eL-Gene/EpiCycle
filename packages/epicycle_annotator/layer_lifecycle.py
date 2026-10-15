"""Backward-compatibility shim — canonical module is subviewer_mirror.py."""

from .subviewer_mirror import SubviewerMirrorManager as LayerLifecycleManager  # noqa: F401

__all__ = ["LayerLifecycleManager"]

