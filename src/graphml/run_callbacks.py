import os
import torch
from typing import Callable, Dict, List
from dataclasses import dataclass
from graphml.metrics import Metric
from .ModelRunner import ModelRunner, EpochStat


class BestMetricAudit(Callable[[ModelRunner, EpochStat], List[bool]]):
    @dataclass
    class BestMetric():
        metric: Metric
        on_epoch: int

    _best_metrics: Dict[str, BestMetric] = {}

    def __init__(self, *metric_selector: Callable[[EpochStat], Metric]):
        self._metrics = metric_selector
        self._counter = 0

    def __call__(self, model_runner: ModelRunner, current_epoch: EpochStat) -> List[bool]:
        metrics_updated = []
        for selector in self._metrics:
            current_metric = selector(current_epoch)
            if current_metric.better_then(self._previous_metric_by_name(current_metric.name)):
                self._update_best_metric(current_metric, current_epoch.epoch)
                metrics_updated.append(True)
            else:
                metrics_updated.append(False)

        self._counter = 0 if any(metrics_updated) else self._counter + 1
        return metrics_updated

    def _previous_metric_by_name(self, name: str) -> Metric:
        return self._best_metrics[name].metric if name in self._best_metrics else None

    def _update_best_metric(self, better_metric: Metric, current_epoch: int):
        self._best_metrics[better_metric.name] = BestMetricAudit.BestMetric(
            better_metric, current_epoch)


class EarlyStopping(BestMetricAudit):
    def __init__(self, patience: int, *metric_selector: Callable[[EpochStat], Metric]):
        super().__init__(*metric_selector)
        self._patience = patience

    def __call__(self, model_runner: ModelRunner, current_epoch: EpochStat) -> List[bool]:
        metrics_updated = super().__call__(model_runner, current_epoch)

        if self._counter >= self._patience:
            model_runner.stop()
            metrics_str = map(
                lambda x: f"Best {str(x.metric)} on epoch {x.on_epoch + 1}", self._best_metrics.values())
            print(
                f"Early stopping on epoch {current_epoch.epoch + 1}: {' '.join(metrics_str)}")

        return metrics_updated


class SaveModelOnBestMetric(BestMetricAudit):
    def __init__(self, path: str, *metric_selector: Callable[[EpochStat], Metric]):
        super().__init__(*metric_selector)
        self._file_path = path

    def __call__(self, model_runner: ModelRunner, current_epoch: EpochStat) -> List[bool]:
        metrics_updated = super().__call__(model_runner, current_epoch)
        if any(metrics_updated):  # any or all?
            torch.save(model_runner._net, self._file_path)

        return metrics_updated
