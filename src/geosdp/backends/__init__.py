from .base import GeoBackend
from .native import NativeGeoBackend
from .reference import ReferenceGeoBackend


def create_backend(name: str) -> GeoBackend:
    normalized = name.strip().lower()
    if normalized == "native":
        return NativeGeoBackend()
    if normalized == "reference":
        return ReferenceGeoBackend()
    raise ValueError(f"unknown backend: {name}")


__all__ = ["GeoBackend", "NativeGeoBackend", "ReferenceGeoBackend", "create_backend"]
