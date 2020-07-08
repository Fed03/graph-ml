import csv
import os
from graphml.paper_nets import GAT_model
from graphml.datasets import CoraDataset, PubmedDataset, CiteseerDataset
from graphml.datasets.Transform import NormalizeFeatures
from graphml.MiniBatchLoader import MiniBatchLoader
from graphml.ModelRunner import ModelRunner, MiniBatchModelRunner
from graphml.run_callbacks import EarlyStopping, SaveModelOnBestMetric

epochs = 100000
patience = 100

current_file_directory = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(current_file_directory, "best_gat.pt")

dataset = CoraDataset(current_file_directory, NormalizeFeatures()).load()

MiniBatchModelRunner(dataset,lambda d: GAT_model(
    d.features_per_node, d.number_of_classes))

""" runner = ModelRunner(dataset, lambda d: GAT_model(
    d.features_per_node, d.number_of_classes))
train_stats = runner.fit(epochs, lambda n, d: n(d.features_vectors, d.adj_coo_matrix),
                         EarlyStopping(patience, lambda x: x.validation_loss, lambda x: x.validation_accuracy), SaveModelOnBestMetric(model_file, lambda x: x.validation_loss, lambda x: x.validation_accuracy))
test_acc, test_loss = runner.test(
    lambda n, d: n(d.features_vectors, d.adj_coo_matrix), model_file)

with open(os.path.join(current_file_directory, "gat_results.csv"), "w", newline="") as csv_file:
    train_stats_dict = map(lambda x: x.asdict(), train_stats)
    train_stats_dict = map(
        lambda x: {**x, "epoch": x["epoch"]+1}, train_stats_dict)
    train_stats_dict = list(map(
        lambda x: {k: v for k, v in x.items() if k != 'total_epochs'}, train_stats_dict))
    writer = csv.DictWriter(csv_file, train_stats_dict[0].keys())
    writer.writeheader()
    writer.writerows(train_stats_dict) """
