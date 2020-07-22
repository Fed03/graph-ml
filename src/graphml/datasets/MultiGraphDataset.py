from dataclasses import dataclass
from typing import List
from .InternalData import GraphData


@dataclass
class MultiGraphDataset():
    train: List[GraphData]
    validation: List[GraphData]
    test: List[GraphData]
    features_per_node: int
    number_of_classes: int

    def to(self, *args, **kwargs):
        self.train = list(
            map(lambda data: data.to(*args, **kwargs), self.train))
        self.validation = list(
            map(lambda data: data.to(*args, **kwargs), self.validation))
        self.test = list(
            map(lambda data: data.to(*args, **kwargs), self.test))
        
        return self
