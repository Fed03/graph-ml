from graphml.layers.sage_aggregators import MeanAggregator, MaxPoolAggregator, ModelSize, LstmAggregator
from utils import write_test_results, write_train_epochs_stats
import os
import torch
from datetime import datetime
from graphml.datasets import PPIDataset
from graphml.paper_nets.GraphSageUnsupervisedNet import GraphSagePPIUnsupervisedModel
from graphml.datasets.Transform import SubSampleNeighborhoodSize, CalcPositivePairs
from graphml.run_callbacks import SaveModelOnBestMetric


def run_sage(aggregator_name):
    epochs = 5
    dataset_name = "ppi"

    datasets = {
        "ppi": PPIDataset
    }
    aggrs = {
        # 0.465
        "gcn": lambda input_size, output_size: MeanAggregator(input_size, output_size),
        # 0.502
        "pool": lambda input_size, output_size: MaxPoolAggregator(input_size, output_size, model_size=ModelSize.SMALL),
        # 0.482
        "lstm": lambda input_size, output_size: LstmAggregator(input_size, output_size, model_size=ModelSize.SMALL)
    }

    model_name = os.path.splitext(os.path.basename(__file__))[0]
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(experiments_dir, model_name,
                             dataset_name, aggregator_name)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = os.path.join(model_dir, run_id)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    model_file = os.path.join(run_dir, f"best_model.pt")
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = datasets[dataset_name](experiments_dir,
                                     SubSampleNeighborhoodSize(128), CalcPositivePairs(50, 5)).load()
    dataset = dataset.to(device)

    model = GraphSagePPIUnsupervisedModel(
        dataset.features_per_node, 256, aggrs[aggregator_name], 1e-5)
    model.to(device)

    train_stats = model.fit(
        epochs,
        dataset.train,
        dataset.validation,
        SaveModelOnBestMetric(
            model_file, lambda x: x.validation_loss)
    )

    write_train_epochs_stats(run_dir, train_stats)

    results = model.test(dataset.test, model_file)
    write_test_results(model_dir, run_id, results)


if __name__ == "__main__":
    for _ in range(10):
        run_sage("gcn")
        run_sage("lstm")
        run_sage("pool")
