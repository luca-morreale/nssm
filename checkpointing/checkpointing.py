
from .binary import BinaryCheckpointing
from .image import ImageCheckpointing
from .surface import SurfaceCheckpointing


class AllCheckpointing(BinaryCheckpointing, ImageCheckpointing, SurfaceCheckpointing):
    pass
