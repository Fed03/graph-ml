from __future__ import annotations
import torch
from time import perf_counter
from graphml.MiniBatchLoader import MiniBatchLoader
from .datasets.InternalData import InternalData
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass, InitVar, field, fields
from .metrics import Loss, Accuracy, Metric
from functools import reduce


def accuracy(logits, labels):
    assert len(logits) == len(labels)
    pred = logits.argmax(dim=1)
    correct_pred_number = torch.eq(pred, labels).sum().item()
    acc = correct_pred_number / len(labels)
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

    def fit(self, epochs: int, run_net: Callable[[torch.nn.Module, torch.Tensor, List[Torch.Tensor]], torch.Tensor], *callbacks: Optional[Callable[[ModelRunner, EpochStat], None]]) -> List[EpochStat]:
        self._total_epochs = epochs
        self._run_net = run_net

        epochs_stats = []
        print("##### Start training #####")
        for epoch in range(epochs):
            stat = self._run_epoch(epoch)
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

    def _run_epoch(self, current_epoch: int) -> EpochStat:
        start = perf_counter()

        train_loss, train_accuracy = self._train()
        validation_accuracy, validation_loss = self._evaluate()

        end = perf_counter()

        return EpochStat(current_epoch, self._total_epochs, train_loss, train_accuracy, validation_loss, validation_accuracy, end-start)

    def _train(self) -> Tuple[float, float]:
        self._net.train()
        self._optimizer.zero_grad()
        output = self._run_net(
            self._net, self._dataset.features_vectors, self._dataset.adj_coo_matrix)
        loss = self._loss_fn(output[self._dataset.train_mask],
                             self._dataset.labels[self._dataset.train_mask])
        loss.backward()
        self._optimizer.step()

        return loss.item(), accuracy(output[self._dataset.train_mask], self._dataset.labels[self._dataset.train_mask])

    def _evaluate(self) -> Tuple[float, float]:
        with torch.no_grad():
            self._net.eval()
            output = self._run_net(
                self._net, self._dataset.features_vectors, self._dataset.adj_coo_matrix)
            validation_accuracy = accuracy(
                output[self._dataset.validation_mask], self._dataset.labels[self._dataset.validation_mask])
            validation_loss = self._loss_fn(
                output[self._dataset.validation_mask], self._dataset.labels[self._dataset.validation_mask]).item()

            return validation_accuracy, validation_loss

    def test(self, best_model_file: Optional[str] = None) -> Tuple[float, float]:
        print("##### Test Model #####")
        with torch.no_grad():
            net = torch.load(best_model_file) if best_model_file else self._net
            net.eval()
            output = self._run_net(
                net, self._dataset.features_vectors, self._dataset.adj_coo_matrix)
            test_accuracy = accuracy(
                output[self._dataset.test_mask], self._dataset.labels[self._dataset.test_mask])
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


class MiniBatchModelRunner(ModelRunner):
    def __init__(self, batch_size: int, dataset: InternalData, model_builder: Callable[[InternalData], Tuple[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.optim.Optimizer]]):
        super().__init__(dataset, model_builder)

        self._train_loader = MiniBatchLoader(
            self._dataset.adj_coo_matrix, self._dataset.train_mask, batch_size=batch_size)

    def _train(self) -> Tuple[float, float]:
        self._net.train()
        losses = []
        accuracies = []
        for target_idxs, input_idxs, adjs in self._train_loader:
            self._optimizer.zero_grad()
            output = self._run_net(
                self._net, self._dataset.features_vectors[input_idxs], *[x.sampled_adj for x in adjs])
            output = self._batch_output_for_target_idx(
                output, target_idxs, input_idxs)
            loss = self._loss_fn(output, self._dataset.labels[target_idxs])
            loss.backward()
            self._optimizer.step()

            losses.append(loss.item())
            accuracies.append(
                accuracy(output, self._dataset.labels[target_idxs]))

        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

    def _batch_output_for_target_idx(self, output, target_idxs, batch_idxs):
        mask = torch.zeros_like(batch_idxs, dtype=torch.bool)
        mask = reduce(lambda msk, trg_idx: msk | (
            batch_idxs == trg_idx), target_idxs, mask)

        return output[mask]
