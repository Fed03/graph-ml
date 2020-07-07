from __future__ import annotations
import torch
from time import perf_counter
from .datasets.InternalData import InternalData
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass, InitVar, field, asdict, fields
from .metrics import Loss, Accuracy, Metric


def accuracy(logits, labels, mask):
    pred = logits[mask].argmax(dim=1)
    correct_pred_number = torch.eq(pred, labels[mask]).sum().item()
    acc = correct_pred_number / mask.sum().item()
    return acc


class ModelRunner():
    def __init__(self, dataset: InternalData, model_builder: Callable[[InternalData], Tuple[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.optim.Optimizer]]):
        self._dataset = dataset.to(self._device)

        self._net, self._loss_fn, self._optimizer = model_builder(
            self._dataset)
        self._net.to(self._device)

        self._stop_requested = False

    @property
    def _device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def fit(self, epochs: int, run_net: Callable[[torch.nn.Module, InternalData], torch.Tensor], *callbacks: Optional[Callable[[ModelRunner, EpochStat], None]]) -> List[EpochStat]:
        self._total_epochs = epochs

        epochs_stats = []
        print("##### Start training #####")
        for epoch in range(epochs):
            stat = self._run_epoch(epoch, run_net)
            print(stat)
            epochs_stats.append(stat)
            for c in callbacks:
                c(self, stat)
            if self._stop_requested:
                break

        print("##### Train ended #####")
        return epochs_stats

    def stop(self):
        self._stop_requested = True

    def _run_epoch(self, current_epoch: int, run_net: Callable[[torch.nn.Module, InternalData], torch.Tensor]) -> EpochStat:
        start = perf_counter()

        train_loss = self._train(run_net)
        train_accuracy, validation_accuracy, validation_loss = self._evaluate(
            run_net)

        end = perf_counter()

        return EpochStat(current_epoch, self._total_epochs, train_loss, train_accuracy, validation_loss, validation_accuracy, end-start)

    def _train(self, run_net: Callable[[torch.nn.Module, InternalData], torch.Tensor]) -> float:
        self._net.train()
        self._optimizer.zero_grad()
        output = run_net(self._net, self._dataset)
        loss = self._loss_fn(output[self._dataset.train_mask],
                             self._dataset.labels[self._dataset.train_mask])
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _evaluate(self, run_net: Callable[[torch.nn.Module, InternalData], torch.Tensor]) -> Tuple[float, float, float]:
        with torch.no_grad():
            self._net.eval()
            output = run_net(self._net, self._dataset)
            train_accuracy = accuracy(
                output, self._dataset.labels, self._dataset.train_mask)
            validation_accuracy = accuracy(
                output, self._dataset.labels, self._dataset.validation_mask)
            validation_loss = self._loss_fn(
                output[self._dataset.validation_mask], self._dataset.labels[self._dataset.validation_mask]).item()

            return train_accuracy, validation_accuracy, validation_loss

    def test(self, run_net: Callable[[torch.nn.Module, InternalData], torch.Tensor], best_model_file: Optional[str] = None) -> Tuple[float, float]:
        print("##### Test Model #####")
        with torch.no_grad():
            net = torch.load(best_model_file) if best_model_file else self._net
            net.eval()
            output = run_net(net, self._dataset)
            test_accuracy = accuracy(
                output, self._dataset.labels, self._dataset.test_mask)
            test_loss = self._loss_fn(
                output[self._dataset.test_mask], self._dataset.labels[self._dataset.test_mask]).item()

            print(
                f"Test Loss {test_loss:.4f}, Test Accuracy {test_accuracy:.4f}")
            return test_loss, test_accuracy


@dataclass
class EpochStat():
    epoch: int
    total_epochs: int
    train_loss: Loss = field(init=False)
    train_loss_val: InitVar[float]
    train_accuracy: Accuracy = field(init=False)
    train_accuracy_val: InitVar[float]
    validation_loss: Loss = field(init=False)
    validation_loss_val: InitVar[float]
    validation_accuracy: Accuracy = field(init=False)
    validation_accuracy_val: InitVar[float]
    elapsed_time: float

    def __post_init__(self, train_loss_val: float, train_accuracy_val: float, validation_loss_val: float, validation_accuracy_val: float):
        self.train_loss = Loss("Train Loss", train_loss_val)
        self.train_accuracy = Accuracy("Train Accuracy", train_accuracy_val)
        self.validation_loss = Loss("Validation Loss", validation_loss_val)
        self.validation_accuracy = Accuracy(
            "Validation Accuracy", validation_accuracy_val)

    def __repr__(self):
        repr_values = map(lambda x: f"{x!r}", self)
        return f"EpochStat({','.join(repr_values)})"

    def __str__(self):
        return f"Epoch {self.epoch + 1}/{self.total_epochs}, {self.train_loss}, {self.train_accuracy}, {self.validation_loss}, {self.validation_accuracy}, Time: {self.elapsed_time:.5f}"

    def asdict(self):
        field_dict = map(lambda x: (
            x.name, getattr(self, x.name)), fields(self))
        return {k: f.value if isinstance(f, Metric) else f for k, f in field_dict}
