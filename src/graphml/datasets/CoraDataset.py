from .PlanetoidDatasetLoader import PlanetoidDatasetLoader
from typing import Callable
from .InternalData import InternalData


class CoraDataset(PlanetoidDatasetLoader):
    def __init__(self, base_path: str, transform: Callable[[InternalData], InternalData] = None):
        super().__init__("cora", base_path, transform)
