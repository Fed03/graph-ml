from __future__ import annotations
from functools import reduce
from itertools import repeat
from time import perf_counter
from typing import Callable, List, Optional, Tuple
from graphml.MiniBatchLoader import BatchStep, MiniBatchLoader
from graphml.ModelRunner import EpochStat
from graphml.datasets.InternalData import GraphData
from graphml.layers.graph_sage import GraphSAGELayer
import torch
import torch.nn.functional as F

from graphml.layers.sage_aggregators import MeanAggregator
from graphml.metrics import Loss, MicroF1


def get_target_idxs_output(net_output: torch.Tensor, batch_step: BatchStep) -> torch.Tensor:
    mask = torch.zeros_like(batch_step.node_idxs,
                            dtype=torch.bool, device=net_output.device)
    mask = reduce(lambda msk, trg_idx: msk | (
        batch_step.node_idxs == trg_idx), batch_step.target_idxs, mask)

    return net_output[mask]


class GraphSageNet(torch.nn.Module):
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int):
        super().__init__()

        self._convs = torch.nn.ModuleList([
            GraphSAGELayer(MeanAggregator(input_feature_dim, 256)),
            GraphSAGELayer(MeanAggregator(256, number_of_classes), lambda x: x)
        ])

    def forward(self, input_matrix: torch.Tensor, *adjs: torch.Tensor):
        assert len(adjs) == len(self._convs) or len(adjs) == 1
        adjs = adjs if len(adjs) != 1 else repeat(adjs[0], len(self._convs))

        x = input_matrix
        for conv, adj in zip(self._convs, adjs):
            x = conv(x, adj)
        return x


class GraphSageSupervisedModel():
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int, learning_rate=.01):
        self._net = GraphSageNet(input_feature_dim, number_of_classes)
        self._loss_fn = F.cross_entropy
        self._optim = torch.optim.Adam(
            self._net.parameters(), learning_rate)
        # pull up
        self._stop_requested = False

    # pull up
    def to(self, *args, **kwargs) -> GraphSageSupervisedModel:
        self._net.to(*args, **kwargs)
        return self

    def fit(self, epochs: int, train_data: GraphData, validation_data: GraphData, *callbacks: Optional[Callable[[GraphSageSupervisedModel, EpochStat], None]]) -> List[EpochStat]:
        self._train_data = train_data
        self._validation_data = validation_data
        self._batch_loader: List[BatchStep] = MiniBatchLoader(
            self._train_data.adj_coo_matrix, self._train_data.train_mask, [25, 10], batch_size=5, shuffle=True)

        return self._internal_fit(epochs, *callbacks)

    # pull up
    def _internal_fit(self, epochs: int, *callbacks: Optional[Callable[[GraphSageSupervisedModel, EpochStat], None]]) -> List[EpochStat]:
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

        train_loss, train_F1 = self._train()
        validation_loss, validation_F1 = self._evaluate()

        end = perf_counter()

        return EpochStat(current_epoch, self._total_epochs, end-start, train_loss, validation_loss, train_F1=train_F1, validation_F1=validation_F1)

    def _train(self) -> Tuple[Loss, MicroF1]:
        self._net.train()

        results = []
        for batch in self._batch_loader:
            self._optim.zero_grad()

            output = self._net(
                self._train_data.features_vectors[batch.node_idxs], *batch.sampled_adj)
            output = get_target_idxs_output(output, batch)

            labels = self._train_data.labels[batch.target_idxs]

            loss = self._loss_fn(output, labels)
            f1 = MicroF1.calc(output, labels)
            loss.backward()
            self._optim.step()

            results.append((loss.item(), f1))

        avg_loss, avg_f1 = self._avg_results(results)
        return Loss("Train Loss", avg_loss), MicroF1("Train F1", avg_f1)

    def _evaluate(self) -> Tuple[Loss, MicroF1]:
        with torch.no_grad():
            self._net.eval()

            output = self._net(self._validation_data.features_vectors, self._validation_data.adj_coo_matrix)

            return Loss("Validation Loss", self._loss_fn(output, self._validation_data.labels).item()), MicroF1("Validation F1", MicroF1.calc(output, self._validation_data.labels))

    def test(self, test_data: GraphData, best_model_file: Optional[str] = None) -> Tuple[Loss, MicroF1]:
        print("##### Test Model #####")
        with torch.no_grad():
            if best_model_file:
                self._net.load_state_dict(torch.load(best_model_file))
            
            self._net.eval()
            output = self._net(test_data.features_vectors, test_data.adj_coo_matrix)

            result = Loss("Test Loss", self._loss_fn(output, test_data.labels).item()), MicroF1("Test F1", MicroF1.calc(output, test_data.labels))

            print(f"{result[0]}, {result[1]}")
            return result

    def _avg_results(self, results: List[Tuple]) -> Tuple:
        sums = [sum(i) for i in zip(*results)]
        return (x/len(results) for x in sums)
