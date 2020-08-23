from utils import write_test_results, write_train_epochs_stats
import os
import torch
from datetime import datetime
from graphml.datasets import PPIDataset
from graphml.paper_nets.GATInductiveNet import GATInductiveModel
from graphml.datasets.Transform import AddSelfLoop, NormalizeFeatures
from graphml.run_callbacks import EarlyStopping, SaveModelOnBestMetric


def run_gat_inductive():
    epochs = 100000
    patience = 100
    dataset_name = "ppi"
    lr = 0.005

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
                                     NormalizeFeatures(), AddSelfLoop()).load()
    dataset = dataset.to(device)

    model = GATInductiveModel(dataset.features_per_node,
                              dataset.number_of_classes, lr)
    model.to(device)

    train_stats = model.fit(
        epochs,
        dataset.train,
        dataset.validation,
        EarlyStopping(
            patience, lambda x: x.validation_loss, lambda x: x.validation_F1),
        SaveModelOnBestMetric(
            model_file, lambda x: x.validation_loss, lambda x: x.validation_F1)
    )

    write_train_epochs_stats(run_dir, train_stats)

    results = model.test(dataset.test, model_file)
    write_test_results(model_dir, run_id, results)


if __name__ == "__main__":
    for _ in range(100):
        run_gat_inductive()
