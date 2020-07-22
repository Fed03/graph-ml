from .PlanetoidDatasetLoader import PlanetoidDatasetLoader
from typing import Callable
from .InternalData import GraphData


class CoraDataset(PlanetoidDatasetLoader):
    def __init__(self, base_path: str, *transform: Callable[[GraphData], GraphData]):
        super().__init__("cora", base_path, *transform)
