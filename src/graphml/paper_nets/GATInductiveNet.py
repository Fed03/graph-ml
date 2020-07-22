from __future__ import annotations
import torch
from time import perf_counter
from collections.abc import Iterable
import torch.nn.functional as F
from itertools import repeat
from typing import Callable, List, Optional, Tuple, Union
from graphml.ModelRunner import EpochStat
from graphml.datasets.InternalData import GraphData
from graphml.layers.gat import MultiHeadGatLayer
from functools import reduce


def accuracy(logits, labels):
    assert len(logits) == len(labels)
    pred = logits.argmax(dim=1)
    correct_pred_number = torch.eq(pred, labels).sum().item()
    acc = correct_pred_number / len(labels)
    return acc


class GATInductiveNet(torch.nn.Module):
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int):
        super().__init__()

        self._convs = torch.nn.ModuleList([
            MultiHeadGatLayer(4, input_feature_dim, 256),
            MultiHeadGatLayer(4, 1024, 256),
            MultiHeadGatLayer(6, 1024, number_of_classes,
                              activation_function=torch.nn.LogSoftmax(dim=1), concat=False)
        ])

    def forward(self, input_matrix: torch.Tensor, *adjs: torch.Tensor):
        assert len(adjs) == len(self._convs) or len(adjs) == 1

        x = input_matrix
        adjs = adjs if len(adjs) != 1 else repeat(adjs[0], len(self._convs))
        for idx, conv, adj in zip(range(len(self._convs)), self._convs, adjs):
            x = conv(x, adj) if idx != 1 else conv(
                x, adj) + x  # skip conn on inter layer
        return x


def GAT_inductive_model(
    input_feature_dim: torch.Size,
    number_of_classes: int,
    learning_rate=.005
) -> Tuple[GATInductiveNet, Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.optim.Optimizer]:
    net = GATInductiveNet(input_feature_dim, number_of_classes)
    return net, F.nll_loss, torch.optim.Adam(net.parameters(), learning_rate)


class GATInductiveModel():
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int, learning_rate=.005):
        self._net = GATInductiveNet(input_feature_dim, number_of_classes)
        self._loss_fn = F.nll_loss
        self._optim = torch.optim.Adam(self._net.parameters(), learning_rate)
        # pull up
        self._stop_requested = False

    # pull up
    def to(self, *args, **kwargs) -> GATInductiveModel:
        self._net.to(*args, **kwargs)
        return self

    def fit(self, epochs: int, train_data: List[GraphData], validation_data: List[GraphData], *callbacks: Optional[Callable[[GATInductiveModel, EpochStat], None]]) -> List[EpochStat]:
        self._train_data = train_data
        self._validation_data = validation_data

        return self._internal_fit(epochs, *callbacks)

    # pull up
    def _internal_fit(self, epochs: int, *callbacks: Optional[Callable[[GATInductiveModel, EpochStat], None]]) -> List[EpochStat]:
        self._total_epochs = epochs
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

    # pull up
    def stop(self):
        self._stop_requested = True

    def test(self, test_data: List[GraphData], best_model_file: Optional[str] = None) -> Tuple[float, float]:
        print("##### Test Model #####")
        with torch.no_grad():
            net = torch.load(best_model_file) if best_model_file else self._net
            net.eval()

            results = []
            for step in test_data:
                output = self._net(step.features_vectors, step.adj_coo_matrix)
                results.append((accuracy(output, step.labels),
                                self._loss_fn(output, step.labels).item()))

            test_accuracy, test_loss = self._avg_results(results)
            print(
                f"Test Loss {test_loss:.4f}, Test Accuracy {test_accuracy:.4f}")
            return test_loss, test_accuracy

    # pull up
    def _run_epoch(self, current_epoch: int) -> EpochStat:
        start = perf_counter()

        train_loss, train_accuracy = self._train()
        validation_accuracy, validation_loss = self._evaluate()

        end = perf_counter()

        return EpochStat(current_epoch, self._total_epochs, train_loss, train_accuracy, validation_loss, validation_accuracy, end-start)

    def _train(self) -> Tuple[float, float]:
        self._net.train()

        results = []
        for step in self._train_data:
            self._optim.zero_grad()
            output = self._net(step.features_vectors, step.adj_coo_matrix)
            loss = self._loss_fn(output, step.labels)
            acc = accuracy(output, step.labels)
            loss.backward()
            self._optim.step()

            results.append((loss.item(), acc))

        return self._avg_results(results)

    def _evaluate(self) -> Tuple[float, float]:
        with torch.no_grad():
            self._net.eval()

            results = []
            for step in self._validation_data:
                output = self._net(step.features_vectors, step.adj_coo_matrix)
                results.append((accuracy(output, step.labels),
                                self._loss_fn(output, step.labels).item()))

            return self._avg_results(results)

    def _avg_results(self, results: List[Tuple]) -> Tuple:
        sums = [sum(i) for i in zip(*results)]
        return (x/len(results) for x in sums)
