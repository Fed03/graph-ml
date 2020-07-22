from .PlanetoidDatasetLoader import PlanetoidDatasetLoader
from typing import Callable
from .InternalData import GraphData


class PubmedDataset(PlanetoidDatasetLoader):
    def __init__(self, base_path: str, *transform: Callable[[GraphData], GraphData]):
        super().__init__("pubmed", base_path, *transform)
