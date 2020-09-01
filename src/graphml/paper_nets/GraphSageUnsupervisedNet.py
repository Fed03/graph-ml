from __future__ import annotations
from itertools import repeat

from graphml.MiniBatchLoader import MiniBatchLoader, BatchStep
from graphml.layers.sage_aggregators import Aggregator
from time import perf_counter
from typing import Callable, List, Optional, Tuple
from graphml.ModelRunner import EpochStat
from graphml.datasets.InternalData import GraphData
from graphml.layers.graph_sage import GraphSAGELayer
import torch
from graphml.metrics import Loss, MicroF1
from graphml.utils import negative_sample_idxs, sample_neighbors
from tqdm import tqdm


class GraphSageNet(torch.nn.Module):
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int, aggregator_factory: Callable[[int, int], Aggregator]):
        super().__init__()

        self._convs = torch.nn.ModuleList([
            GraphSAGELayer(aggregator_factory(input_feature_dim, 256)),
            GraphSAGELayer(aggregator_factory(
                256, number_of_classes), lambda x: x)
        ])

    def forward(self, input_matrix: torch.Tensor, *adjs):
        assert len(adjs) == len(self._convs) or len(adjs) == 1
        adjs = adjs if len(adjs) != 1 else repeat(adjs[0], len(self._convs))

        x = input_matrix
        for conv, adj in zip(self._convs, adjs):
            x = conv(x, adj)
        return x


class GraphSagePPIUnsupervisedModel():
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int, aggregator_factory: Callable[[int, int], Aggregator], learning_rate=.01):
        self._net = GraphSageNet(
            input_feature_dim, number_of_classes, aggregator_factory)
        self._loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self._optim = torch.optim.Adam(
            self._net.parameters(), learning_rate)
        # pull up
        self._stop_requested = False

    # pull up
    def to(self, *args, **kwargs) -> GraphSagePPIUnsupervisedModel:
        self._net.to(*args, **kwargs)
        return self

    def fit(self, epochs: int, train_data: List[GraphData], validation_data: List[GraphData], *callbacks: Optional[Callable[[GraphSagePPIUnsupervisedModel, EpochStat], None]]) -> List[EpochStat]:
        self._train_data = train_data
        self._validation_data = validation_data

        return self._internal_fit(epochs, *callbacks)

    # pull up
    def _internal_fit(self, epochs: int, *callbacks: Optional[Callable[[GraphSagePPIUnsupervisedModel, EpochStat], None]]) -> List[EpochStat]:
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
        for train_graph in self._train_data:
            self._optim.zero_grad()
            adjs = [sample_neighbors(train_graph.adj_coo_matrix,size) for size in [25,10]]
            output = self._net(train_graph.features_vectors,
                               *adjs)
            labels = train_graph.labels

            loss = self.unsup_loss(
                output, train_graph.positive_pairs, train_graph.adj_coo_matrix)
            f1 = MicroF1.calc(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self._net.parameters(), 5)
            self._optim.step()

            results.append((loss.item(), f1))

        avg_loss, avg_f1 = self._avg_results(results)
        return Loss("Train Loss", avg_loss), MicroF1("Train F1", avg_f1)

    def _evaluate(self) -> Tuple[Loss, MicroF1]:
        with torch.no_grad():
            self._net.eval()

            results = []
            for graph in self._validation_data:
                adjs = [sample_neighbors(graph.adj_coo_matrix,size) for size in [25,10]]
                output = self._net(graph.features_vectors,
                                   *adjs)
                labels = graph.labels

                loss = self.unsup_loss(
                    output, graph.positive_pairs, graph.adj_coo_matrix)
                f1 = MicroF1.calc(output, labels)

                results.append((loss.item(), f1))

            avg_loss, avg_f1 = self._avg_results(results)

            return Loss("Validation Loss", avg_loss), MicroF1("Validation F1", avg_f1)

    def test(self, test_data: List[GraphData], best_model_file: Optional[str] = None) -> Tuple[Loss, MicroF1]:
        print("##### Test Model #####")
        with torch.no_grad():
            if best_model_file:
                self._net.load_state_dict(torch.load(best_model_file))

            self._net.eval()
            results = []
            for graph in test_data:
                adjs = [sample_neighbors(graph.adj_coo_matrix,size) for size in [25,10]]
                output = self._net(graph.features_vectors,
                                   *adjs)
                labels = graph.labels

                loss = self.unsup_loss(
                    output, graph.positive_pairs, graph.adj_coo_matrix)
                f1 = MicroF1.calc(output, labels)

                results.append((loss.item(), f1))

            avg_loss, avg_f1 = self._avg_results(results)

            result = Loss("Test Loss", avg_loss), MicroF1("Test F1", avg_f1)

            print(f"{result[0]}, {result[1]}")
            return result

    def _avg_results(self, results: List[Tuple]) -> Tuple:
        sums = [sum(i) for i in zip(*results)]
        return (x/len(results) for x in sums)

    def unsup_loss(self, output: torch.Tensor, positive_pairs_idxs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        positive_output = output[positive_pairs_idxs]

        if len(positive_output) != len(output):
            nodes_with_positive = adj[0].unique()
            tmp = torch.zeros(len(output), positive_output.size(
                1), positive_output.size(2), device=positive_output.device)
            tmp[nodes_with_positive] = positive_output
            positive_output = tmp

        positive_value = torch.matmul(
            positive_output, output.unsqueeze(2)).sum(1).squeeze()

        neg_idxs = negative_sample_idxs(20, adj)
        negative_values = torch.matmul(
            output[neg_idxs], output.unsqueeze(2)).squeeze()

        return self._loss_fn(positive_value, torch.ones_like(positive_value)) + self._loss_fn(negative_values, torch.zeros_like(negative_values))

class GraphSageRedditUnupervisedModel():
    def __init__(self, input_feature_dim: torch.Size, number_of_classes: int,aggregator_factory, learning_rate=.01):
        self._net = GraphSageNet(input_feature_dim, number_of_classes,aggregator_factory)
        self._loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self._optim = torch.optim.Adam(
            self._net.parameters(), learning_rate)
        # pull up
        self._stop_requested = False
        self._current_device = torch.device("cpu")

    # pull up
    def to(self, device: torch.device) -> GraphSageRedditUnupervisedModel:
        self._current_device = device
        self._net.to(device)
        return self

    def fit(self, epochs: int, train_data: GraphData, validation_data: GraphData, *callbacks: Optional[Callable[[GraphSageRedditUnupervisedModel, EpochStat], None]]) -> List[EpochStat]:
        self._train_data = train_data
        self._validation_data = validation_data
        self._train_loader: List[BatchStep] = MiniBatchLoader(
            self._train_data.adj_coo_matrix, [25, 10], self._train_data.train_mask, batch_size=512, shuffle=True)
        self._validation_loader: List[BatchStep] = MiniBatchLoader(
            self._validation_data.adj_coo_matrix, [25, 10], self._validation_data.validation_mask, batch_size=512, shuffle=False)

        return self._internal_fit(epochs, *callbacks)

    # pull up
    def _internal_fit(self, epochs: int, *callbacks: Optional[Callable[[GraphSageRedditUnupervisedModel, EpochStat], None]]) -> List[EpochStat]:
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
        for batch_step in tqdm(self._train_loader):
            batch_step = batch_step.to(self._current_device)
            self._optim.zero_grad()
            output = self._net(
                self._train_data.features_vectors[batch_step.batch_idxs].to(self._current_device), *batch_step.sampled_adjs)
            output = output[batch_step.sampled_idxs]
            labels = self._train_data.labels[batch_step.target_idxs].to(self._current_device)

            loss = self.unsup_loss(
                    output, self._train_data.positive_pairs, self._train_data.adj_coo_matrix)
            f1 = MicroF1.calc(output, labels)
            loss.backward()
            self._optim.step()

            results.append((loss.item(), f1))

        avg_loss, avg_f1 = self._avg_results(results)
        return Loss("Train Loss", avg_loss), MicroF1("Train F1", avg_f1)

    def _evaluate(self) -> Tuple[Loss, MicroF1]:
        with torch.no_grad():
            self._net.eval()

            results = []
            for batch_step in tqdm(self._validation_loader):
                batch_step = batch_step.to(self._current_device)
                output = self._net(
                    self._validation_data.features_vectors[batch_step.batch_idxs].to(self._current_device), *batch_step.sampled_adjs)
                output = output[batch_step.sampled_idxs]
                labels = self._validation_data.labels[batch_step.target_idxs].to(self._current_device)

                loss = self.unsup_loss(
                    output, self._validation_data.positive_pairs, self._validation_data.adj_coo_matrix)
                f1 = MicroF1.calc(output, labels)

                results.append((loss.item(), f1))

            avg_loss, avg_f1 = self._avg_results(results)

            return Loss("Validation Loss", avg_loss), MicroF1("Validation F1", avg_f1)

    def test(self, test_data: GraphData, best_model_file: Optional[str] = None) -> Tuple[Loss, MicroF1]:
        print("##### Test Model #####")
        loader = MiniBatchLoader(
            test_data.adj_coo_matrix, [25,10], test_data.test_mask, batch_size=512, shuffle=False)
        with torch.no_grad():
            if best_model_file:
                self._net.load_state_dict(torch.load(best_model_file))

            self._net.eval()
            results = []
            for batch_step in tqdm(loader):
                batch_step = batch_step.to(self._current_device)
                output = self._net(
                    test_data.features_vectors[batch_step.batch_idxs].to(self._current_device), *batch_step.sampled_adjs)
                output = output[batch_step.sampled_idxs]
                labels = test_data.labels[batch_step.target_idxs].to(self._current_device)

                loss = self.unsup_loss(
                    output, test_data.positive_pairs, test_data.adj_coo_matrix)
                f1 = MicroF1.calc(output, labels)

                results.append((loss.item(), f1))

            avg_loss, avg_f1 = self._avg_results(results)

            result = Loss("Test Loss", avg_loss), MicroF1("Test F1", avg_f1)

            print(f"{result[0]}, {result[1]}")
            return result

    def _avg_results(self, results: List[Tuple]) -> Tuple:
        sums = [sum(i) for i in zip(*results)]
        return (x/len(results) for x in sums)

    def unsup_loss(self, output: torch.Tensor, positive_pairs_idxs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        positive_output = output[positive_pairs_idxs]

        if len(positive_output) != len(output):
            nodes_with_positive = adj[0].unique()
            tmp = torch.zeros(len(output), positive_output.size(
                1), positive_output.size(2), device=positive_output.device)
            tmp[nodes_with_positive] = positive_output
            positive_output = tmp

        positive_value = torch.matmul(
            positive_output, output.unsqueeze(2)).sum(1).squeeze()

        neg_idxs = negative_sample_idxs(20, adj)
        negative_values = torch.matmul(
            output[neg_idxs], output.unsqueeze(2)).squeeze()

        return self._loss_fn(positive_value, torch.ones_like(positive_value)) + self._loss_fn(negative_values, torch.zeros_like(negative_values))