from utils import write_test_results, write_train_epochs_stats
import os
import torch
from datetime import datetime
from graphml.datasets import PPIDataset
from graphml.paper_nets.GraphSageNet import GraphSagePPISupervisedModel
from graphml.datasets.Transform import AddSelfLoop, NormalizeFeatures, SubSampleNeighborhoodSize
from graphml.run_callbacks import EarlyStopping, SaveModelOnBestMetric


def run_sage():
    epochs = 10
    dataset_name = "ppi"

    datasets = {
        "ppi": PPIDataset  # 0.973 ± 0.002 micro F1
    }
    model_name = os.path.splitext(os.path.basename(__file__))[0]
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(experiments_dir, model_name, dataset_name)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = os.path.join(model_dir, run_id)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    model_file = os.path.join(run_dir, f"best_model.pt")
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = datasets[dataset_name](experiments_dir,
                                     SubSampleNeighborhoodSize(128)).load()
    dataset = dataset.to(device)

    model = GraphSagePPISupervisedModel(dataset.features_per_node,dataset.number_of_classes)
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
    run_sage()
    """ for _ in range(10):
        run_gat_inductive() """
