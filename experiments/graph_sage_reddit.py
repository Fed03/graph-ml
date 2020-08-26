from __future__ import annotations
import os
import torch
from datetime import datetime
from graphml.layers.sage_aggregators import MeanAggregator, MaxPoolAggregator, LstmAggregator, ModelSize

from graphml.paper_nets.GraphSageNet import GraphSageRedditSupervisedModel
from graphml.datasets.InternalData import GraphData
from utils import write_test_results, write_train_epochs_stats
from graphml.datasets.Transform import SubSampleNeighborhoodSize
from graphml.run_callbacks import EarlyStopping, SaveModelOnBestMetric
from graphml.datasets import PPIDataset, RedditDataset


def run_sage(dataset_name, aggr_name):
    epochs = 10
    #patience = 10

    datasets = {
        "ppi": PPIDataset,
        "reddit": RedditDataset
    }

    aggrs = {
        # 0.930
        "gcn": lambda input_size, output_size: MeanAggregator(input_size, output_size),
        # 0.948
        "pool": lambda input_size, output_size: MaxPoolAggregator(input_size, output_size, model_size=ModelSize.SMALL),
        # 0.954
        "lstm": lambda input_size, output_size: LstmAggregator(input_size, output_size, model_size=ModelSize.SMALL)
    }

    model_name = os.path.splitext(os.path.basename(__file__))[0]
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(
        experiments_dir, model_name, dataset_name, aggr_name)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = os.path.join(model_dir, run_id)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    model_file = os.path.join(run_dir, f"best_model.pt")
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = datasets[dataset_name](
        experiments_dir,
        SubSampleNeighborhoodSize(128)
    ).load()

    model = GraphSageRedditSupervisedModel(
        dataset.features_per_node,
        dataset.number_of_classes,
        aggregator_factory=aggrs[aggr_name]
    )
    model.to(device)

    train_data = dataset
    validation_data = dataset
    test_data = dataset
    train_stats = model.fit(
        epochs,
        train_data,
        validation_data,
        SaveModelOnBestMetric(model_file, lambda x: x.validation_loss)
    )

    write_train_epochs_stats(run_dir, train_stats)

    results = model.test(test_data, model_file)
    write_test_results(model_dir, run_id, results)


if __name__ == "__main__":
    run_sage("reddit", "lstm")
