from __future__ import annotations
import torch
import torch.nn.functional as F
from time import perf_counter
from typing import Callable, List, Optional, Tuple
from graphml.datasets.InternalData import GraphData
from graphml.ModelRunner import EpochStat
from graphml.layers.gcn import GCNLayerFactory
from graphml.metrics import Metric

from graphml.metrics import Accuracy

from graphml.metrics import Loss


class GCNNet(torch.nn.Module):
    def __init__(self, adjency_coo_matrix: torch.Tensor, input_feature_dim: torch.Size, number_of_classes: int):
        super().__init__()

        layer_factory = GCNLayerFactory(adjency_coo_matrix)
        self._conv1 = layer_factory.create(input_feature_dim, 16)
        self._conv2 = layer_factory.create(
            16, number_of_classes, torch.nn.LogSoftmax(dim=1), .5)

    def forward(self, input_matrix: torch.Tensor):
        x = self._conv1(input_matrix)
        x = self._conv2(x)
        return x

def GCN_model(
    adjency_coo_matrix: torch.Tensor,
    input_feature_dim: torch.Size,
    number_of_classes: int,
    learning_rate=.01,
    weight_decay=5e-4
) -> Tuple[GCNNet, Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.optim.Optimizer]:
    net = GCNNet(adjency_coo_matrix, input_feature_dim, number_of_classes)
    return net, F.nll_loss, torch.optim.Adam(net.parameters(), learning_rate,weight_decay=weight_decay)


class GCNModel():
    def __init__(self, adjency_coo_matrix: torch.Tensor,input_feature_dim: torch.Size, number_of_classes: int, learning_rate=.01, weight_decay=5e-4):
        self._net = GCNNet(adjency_coo_matrix, input_feature_dim, number_of_classes)
        self._loss_fn = F.nll_loss
        self._optim = torch.optim.Adam(
            self._net.parameters(), learning_rate, weight_decay=weight_decay)
        # pull up
        self._stop_requested = False

    # pull up
    def to(self, *args, **kwargs) -> GCNModel:
        self._net.to(*args, **kwargs)
        return self

    def fit(self, epochs: int, train_data: GraphData, validation_data: GraphData, *callbacks: Optional[Callable[[GCNModel, EpochStat], None]]) -> List[EpochStat]:
        self._train_data = train_data
        self._validation_data = validation_data

        return self._internal_fit(epochs, *callbacks)

    # pull up
    def _internal_fit(self, epochs: int, *callbacks: Optional[Callable[[GCNModel, EpochStat], None]]) -> List[EpochStat]:
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

    # pull up
    def _run_epoch(self, current_epoch: int) -> EpochStat:
        start = perf_counter()

        train_loss, train_accuracy = self._train()
        validation_loss, validation_accuracy = self._evaluate()

        end = perf_counter()

        return EpochStat(current_epoch, self._total_epochs, end-start, train_loss, validation_loss, train_accuracy=train_accuracy, validation_accuracy=validation_accuracy)

    def _train(self) -> Tuple[Metric, Metric]:
        self._net.train()

        self._optim.zero_grad()
        output = self._net(self._train_data.features_vectors)
        loss = self._loss_fn(
            output[self._train_data.train_mask], self._train_data.labels)
        acc = Accuracy.calc(
            output[self._train_data.train_mask], self._train_data.labels)
        loss.backward()
        self._optim.step()

        return Loss("Train Loss", loss.item()), Accuracy("Train Accuracy", acc)

    def _evaluate(self) -> Tuple[Metric, Metric]:
        with torch.no_grad():
            self._net.eval()

            output = self._net(
                self._validation_data.features_vectors)
            loss = self._loss_fn(
                output[self._validation_data.validation_mask], self._validation_data.labels)
            acc = Accuracy.calc(
                output[self._validation_data.validation_mask], self._validation_data.labels)

            return Loss("Validation Loss", loss.item()), Accuracy("Validation Accuracy", acc)

    def test(self, test_data: GraphData, best_model_file: Optional[str] = None) -> Tuple[Metric, Metric]:
        print("##### Test Model #####")
        with torch.no_grad():
            if best_model_file:
                self._net.load_state_dict(torch.load(best_model_file))
            self._net.eval()

            output = self._net(test_data.features_vectors)
            loss = self._loss_fn(output[test_data.test_mask], test_data.labels)
            acc = Accuracy.calc(output[test_data.test_mask], test_data.labels)

            results = Loss("Test Loss", loss.item()), Accuracy(
                "Test Accuracy", acc)

            print(f"{results[0]}, {results[1]}")
            return results
