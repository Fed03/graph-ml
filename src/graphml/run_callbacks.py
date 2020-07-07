import os
import torch
from typing import Callable, Dict
from dataclasses import dataclass
from graphml.metrics import Metric
from .ModelRunner import ModelRunner, EpochStat


class BestMetricAudit(Callable[[ModelRunner, EpochStat], None]):
    @dataclass
    class BestMetric():
        metric: Metric
        on_epoch: int

    _best_metrics: Dict[str, BestMetric] = {}

    def __init__(self, *metric_selector: Callable[[EpochStat], Metric]):
        self._metrics = metric_selector
        self._counter = 0

    def __call__(self, model_runner: ModelRunner, current_epoch: EpochStat):
        self._counter += 1
        for selector in self._metrics:
            current_metric = selector(current_epoch)
            if current_metric.better_then(self._previous_metric_by_name(current_metric.name)):
                self._update_best_metric(current_metric, current_epoch.epoch)
                self._counter = 0

    def _previous_metric_by_name(self, name: str) -> Metric:
        return self._best_metrics[name].metric if name in self._best_metrics else None

    def _update_best_metric(self, better_metric: Metric, current_epoch: int):
        self._best_metrics[better_metric.name] = BestMetricAudit.BestMetric(
            better_metric, current_epoch)


class EarlyStopping(BestMetricAudit):
    def __init__(self, patience: int, *metric_selector: Callable[[EpochStat], Metric]):
        super().__init__(*metric_selector)
        self._patience = patience

    def __call__(self, model_runner: ModelRunner, current_epoch: EpochStat):
        super().__call__(model_runner, current_epoch)

        if self._counter >= self._patience:
            model_runner.stop()
            metric = self._get_oldest_updated_metric()
            print(
                f"Early stopping on {metric.metric.name} on epoch {current_epoch.epoch + 1}")

    def _get_oldest_updated_metric(self) -> BestMetricAudit.BestMetric:
        return sorted(self._best_metrics.values(), key=lambda m: m.on_epoch)[0]


class SaveModelOnBestMetric(BestMetricAudit):
    def __init__(self, path: str, *metric_selector: Callable[[EpochStat], Metric]):
        super().__init__(*metric_selector)
        self._file_path = path

    def __call__(self, model_runner: ModelRunner, current_epoch: EpochStat):
        super().__call__(model_runner, current_epoch)
        if self._counter == 0:
            torch.save(model_runner._net, self._file_path)
