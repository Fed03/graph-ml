from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from sklearn.metrics import f1_score

import torch


""" class MetricCalculator(ABC):
    @abstractmethod
    def calc(self, prefix: str, logits: torch.Tensor, labels: torch.Tensor) -> Metric:
        pass """


@dataclass
class Metric(ABC):
    name: str
    value: float

    @abstractmethod
    def better_then(self, other: Metric):
        ...

    def __str__(self):
        return f"{self.name}: {self.value:.4f}"


@dataclass
class Loss(Metric):
    def better_then(self, other: Loss):
        return True if other is None else self.value <= other.value

    """ class Calculator(MetricCalculator):
        def __init__(self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            self._loss_fn = loss_fn

        def calc(self, name: str, logits: torch.Tensor, labels: torch.Tensor) -> Metric:
            return Loss(f"{name}_loss", self._loss_fn(logits, labels).item()) """


@dataclass
class Accuracy(Metric):
    def better_then(self, other: Accuracy):
        return True if other is None else self.value >= other.value

    @staticmethod
    def calc(logits: torch.Tensor, labels: torch.Tensor) -> float:
        assert len(logits) == len(labels)
        pred = logits.argmax(dim=1)
        correct_pred_number = torch.eq(pred, labels).sum().item()
        acc = correct_pred_number / len(labels)
        return acc 

    """ class Calculator(MetricCalculator):
        def calc(self, name: str, logits: torch.Tensor, labels: torch.Tensor) -> Metric:
            return Accuracy(f"{name}_loss", self.accuracy(logits, labels))

        @staticmethod
        def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
            assert len(logits) == len(labels)
            pred = logits.argmax(dim=1)
            correct_pred_number = torch.eq(pred, labels).sum().item()
            acc = correct_pred_number / len(labels)
            return acc """


@dataclass
class MicroF1(Metric):
    def better_then(self, other: MicroF1):
        return True if other is None else self.value >= other.value

    @staticmethod
    def calc(logits: torch.Tensor, labels: torch.Tensor) -> float:
        pred = logits.detach().sigmoid().round()
        return f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='micro')

    """ class Calculator(MetricCalculator):
        def calc(self, name: str, logits: torch.Tensor, labels: torch.Tensor) -> Metric:
            return MicroF1(f"{name}_loss", self.micro_f1(logits, labels))

        @staticmethod
        def micro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
            pred = logits.sigmoid().round()
            return f1_score(labels, pred, average='micro') """
