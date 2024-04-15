from typing import TYPE_CHECKING, Dict, Type
from . import CameraBase

class CameraFactory:
    def __init__(self, registry: Dict[str, Type["CameraBase"]]):
        self.registry = registry

    def __call__(self, camera_name: str, *args, **kwargs):
        try:
            return self.registry[camera_name.lower()](*args, **kwargs)
        except KeyError:
            raise KeyError(f"No camera with name {camera_name} found. Ensure that any camera clases added inherit from CameraBase")

Camera = CameraFactory(CameraBase._registry)
