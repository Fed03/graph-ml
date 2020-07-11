from __future__ import annotations
import os
from graphml.ModelRunner import ModelRunner
from graphml.run_callbacks import EarlyStopping, SaveModelOnBestMetric
from graphml.paper_nets import GCN_model
from graphml.datasets import CoraDataset, PubmedDataset, CiteseerDataset
from graphml.datasets.Transform import AddSelfLoop, NormalizeFeatures
import csv

model = "gcn"
epochs = 200
patience = 10

current_file_directory = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(current_file_directory, f"best_{model}.pt")

dataset = CoraDataset(current_file_directory, NormalizeFeatures(), AddSelfLoop()).load()

runner = ModelRunner(dataset, lambda d: GCN_model(
    d.adj_coo_matrix, d.features_per_node, d.number_of_classes))
train_stats = runner.fit(epochs, lambda net, x, _: net(x),
                         EarlyStopping(patience, lambda x: x.validation_loss), SaveModelOnBestMetric(model_file, lambda x: x.validation_loss))
test_acc, test_loss = runner.test(
    lambda n, d: n(d.features_vectors), model_file)

with open(os.path.join(current_file_directory, "results.csv"), "w", newline="") as csv_file:
    train_stats_dict = map(lambda x: x.asdict(), train_stats)
    train_stats_dict = map(
        lambda x: {**x, "epoch": x["epoch"]+1}, train_stats_dict)
    train_stats_dict = list(map(
        lambda x: {k: v for k, v in x.items() if k != 'total_epochs'}, train_stats_dict))
    writer = csv.DictWriter(csv_file, train_stats_dict[0].keys())
    writer.writeheader()
    writer.writerows(train_stats_dict)
