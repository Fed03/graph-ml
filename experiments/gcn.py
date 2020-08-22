from __future__ import annotations
import os
import torch
from datetime import datetime
from graphml.paper_nets.GCNNet import GCNModel
from graphml.datasets.InternalData import GraphData
from utils import write_test_results, write_train_epochs_stats
from graphml.datasets.Transform import AddSelfLoop, NormalizeFeatures
from graphml.run_callbacks import EarlyStopping, SaveModelOnBestMetric
from graphml.datasets import CoraDataset, PubmedDataset, CiteseerDataset


def run_gcn(dataset_name):
    epochs = 200
    patience = 10

    datasets = {
        "cora": CoraDataset,  # 81.5%
        "pubmed": PubmedDataset,  # 79.0%
        "citeseer": CiteseerDataset  # 70.3%
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

    model = GCNModel(
        dataset.adj_coo_matrix,
        dataset.features_per_node,
        dataset.number_of_classes
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
        EarlyStopping(patience, lambda x: x.validation_loss),
        SaveModelOnBestMetric(model_file, lambda x: x.validation_loss)
    )

    write_train_epochs_stats(run_dir, train_stats)

    results = model.test(test_data, model_file)
    write_test_results(model_dir, run_id, results)


if __name__ == "__main__":
    for _ in range(100):
        # Citeseer
        run_gcn("citeseer")
        print("Finished Citeseer")
        # Cora
        run_gcn("cora")
        print("Finished Cora")
        # Pubmed
        run_gcn("pubmed")
        print("Finished Pubmed")
