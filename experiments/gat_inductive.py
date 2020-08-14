import os
import csv
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
    lr = 0.005  # 0.01 for pubmed, 0.005 for the others

    datasets = {
        "ppi": PPIDataset
    }
    current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, current_file_name, dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    model_file = os.path.join(
        save_dir, f"best_{current_time}.pt")
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")


    dataset = datasets[dataset_name](current_dir,
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
    test_results = model.test(dataset.test, model_file)

    with open(os.path.join(save_dir, f"results_{current_time}.csv"), "w", newline="") as csv_file:
        train_stats_dict = map(lambda x: x.asdict(), train_stats)
        train_stats_dict = map(
            lambda x: {**x, "epoch": x["epoch"]+1}, train_stats_dict)
        train_stats_dict = list(map(
            lambda x: {k: v for k, v in x.items() if k != 'total_epochs'}, train_stats_dict))
        writer = csv.DictWriter(csv_file, train_stats_dict[0].keys())
        writer.writeheader()
        writer.writerows(train_stats_dict)
    
    return test_results

if __name__ == "__main__":
    test_acc, test_loss = run_gat_inductive()