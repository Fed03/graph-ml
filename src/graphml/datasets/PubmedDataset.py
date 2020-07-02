from .PlanetoidDatasetLoader import PlanetoidDatasetLoader
from typing import Callable
from .InternalData import InternalData


class PubmedDataset(PlanetoidDatasetLoader):
    def __init__(self, base_path: str, transform: Callable[[InternalData], InternalData] = None):
        super().__init__("pubmed", base_path, transform)
