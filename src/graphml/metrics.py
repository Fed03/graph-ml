from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Metric(ABC):
    name: str
    value: float

    @abstractmethod
    def better_then(self, other: Metric):
        ...

    def __str__(self):
        return f"{self.name} {self.value:.4f}"


@dataclass
class Loss(Metric):
    def better_then(self, other: Metric):
        return True if other is None else self.value <= other.value


@dataclass
class Accuracy(Metric):
    def better_then(self, other: Metric):
        return True if other is None else self.value >= other.value
