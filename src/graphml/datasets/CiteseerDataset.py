from .PlanetoidDatasetLoader import PlanetoidDatasetLoader
from typing import Callable
from .InternalData import InternalData


class CiteseerDataset(PlanetoidDatasetLoader):
    def __init__(self, base_path: str, *transform: Callable[[InternalData], InternalData]):
        super().__init__("citeseer", base_path, *transform)
