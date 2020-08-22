import os
import torch
from datetime import datetime
from graphml.datasets.InternalData import GraphData
from utils import write_test_results, write_train_epochs_stats
from graphml.datasets.Transform import AddSelfLoop, NormalizeFeatures
from graphml.run_callbacks import EarlyStopping, SaveModelOnBestMetric
from graphml.paper_nets.GATTransductiveNet import GATTransductiveModel
from graphml.datasets import CoraDataset, PubmedDataset, CiteseerDataset


def run_gat_transductive(dataset_name, lr):
    epochs = 100000
    patience = 100

    datasets = {
        "cora": CoraDataset,  # 83.0 ± 0.7%
        "pubmed": PubmedDataset,  # 79.0 ± 0.3%
        "citeseer": CiteseerDataset  # 72.5 ± 0.7%
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

    dataset = datasets[dataset_name](
        experiments_dir,
        NormalizeFeatures(),
        AddSelfLoop()
    ).load()
    dataset = dataset.to(device)

    model = GATTransductiveModel(
        dataset.features_per_node,
        dataset.number_of_classes,
        lr
    )
    model.to(device)

    train_data = GraphData(
        dataset.name,
        dataset.features_vectors,
        dataset.labels[dataset.train_mask],
        dataset.adj_coo_matrix,
        train_mask=dataset.train_mask
    )
    validation_data = GraphData(
        dataset.name,
        dataset.features_vectors,
        dataset.labels[dataset.validation_mask],
        dataset.adj_coo_matrix,
        validation_mask=dataset.validation_mask
    )
    test_data = GraphData(
        dataset.name,
        dataset.features_vectors,
        dataset.labels[dataset.test_mask],
        dataset.adj_coo_matrix,
        test_mask=dataset.test_mask
    )
    train_stats = model.fit(
        epochs,
        train_data,
        validation_data,
        EarlyStopping(
            patience, lambda x: x.validation_loss, lambda x: x.validation_accuracy),
        SaveModelOnBestMetric(
            model_file, lambda x: x.validation_loss, lambda x: x.validation_accuracy)
    )
    write_train_epochs_stats(run_dir, train_stats)

    results = model.test(test_data, model_file)
    write_test_results(model_dir, run_id, results)


if __name__ == "__main__":
    for _ in range(100):
        # Citeseer
        run_gat_transductive("citeseer", 0.005)
        print("Finished Citeseer")
        # Cora
        run_gat_transductive("cora", 0.005)
        print("Finished Cora")
        # Pubmed
        run_gat_transductive("pubmed", 0.01)
        print("Finished Pubmed")
