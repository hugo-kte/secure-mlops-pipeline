"""Hardened inference API: verify the signed bundle, then serve it read-only."""

from mlseal.serving.app import create_app
from mlseal.serving.settings import Settings

__all__ = ["Settings", "create_app"]
