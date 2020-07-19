import csv
import os
from datetime import datetime
from graphml.paper_nets import GAT_inductive_model
from graphml.datasets import PPIDataset
from graphml.datasets.Transform import AddSelfLoop, NormalizeFeatures
from graphml.ModelRunner import ModelRunner
from graphml.run_callbacks import EarlyStopping, SaveModelOnBestMetric

epochs = 100000
patience = 100
dataset_name = "citeseer"
lr = 0.005  # 0.01 for pubmed, 0.005 for the others

datasets = {
    "cora": CoraDataset,  # 83.0 ± 0.7%
    "pubmed": PubmedDataset,  # 79.0 ± 0.3%
    "citeseer": CiteseerDataset  # 72.5 ± 0.7%
}
current_file_name = os.path.splitext(os.path.basename(__file__))[0]
current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, current_file_name, dataset_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
model_file = os.path.join(
    save_dir, f"best_{current_time}.pt")


dataset = datasets[dataset_name](current_dir,
                                 NormalizeFeatures(), AddSelfLoop()).load()

runner = ModelRunner(dataset, lambda d: GAT_inductive_model(
    d.features_per_node, d.number_of_classes, lr))
train_stats = runner.fit(
    epochs,
    lambda net, input, adjs: net(input, adjs),
    EarlyStopping(
        patience, lambda x: x.validation_loss, lambda x: x.validation_accuracy),
    SaveModelOnBestMetric(
        model_file, lambda x: x.validation_loss, lambda x: x.validation_accuracy)
)
test_acc, test_loss = runner.test(model_file)

with open(os.path.join(save_dir, f"results_{current_time}.csv"), "w", newline="") as csv_file:
    train_stats_dict = map(lambda x: x.asdict(), train_stats)
    train_stats_dict = map(
        lambda x: {**x, "epoch": x["epoch"]+1}, train_stats_dict)
    train_stats_dict = list(map(
        lambda x: {k: v for k, v in x.items() if k != 'total_epochs'}, train_stats_dict))
    writer = csv.DictWriter(csv_file, train_stats_dict[0].keys())
    writer.writeheader()
    writer.writerows(train_stats_dict)
