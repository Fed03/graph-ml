import os
import math
import torch
from graphml.paper_nets import GCN_model
from graphml.datasets.Cora import CoraDataset
from graphml.datasets.Transform import NormalizeFeatures

epochs = 200

current_file_directory = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(current_file_directory, "best_gcn.pt")
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset = CoraDataset(current_file_directory, NormalizeFeatures())
dataset.to(dev)

net, loss_fn, optimizer = GCN_model(
    dataset.adj_coo_matrix, dataset.features_per_node, dataset.number_of_classes)
net.to(dev)


def train():
    net.train()
    optimizer.zero_grad()
    output = net(dataset.features_vectors)
    loss = loss_fn(output[dataset.train_mask],
                   dataset.labels[dataset.train_mask])
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def evaluate():
    net.eval()
    output = net(dataset.features_vectors)
    train_accuracy = accuracy(output, dataset.labels, dataset.train_mask)
    validation_accuracy = accuracy(
        output, dataset.labels, dataset.validation_mask)
    validation_loss = loss_fn(
        output[dataset.validation_mask], dataset.labels[dataset.validation_mask])

    return train_accuracy, validation_accuracy, validation_loss


@torch.no_grad()
def test():
    model_test = torch.load(model_file)
    model_test.eval()
    output = model_test(dataset.features_vectors)
    test_accuracy = accuracy(
        output, dataset.labels, dataset.test_mask)
    test_loss = loss_fn(
        output[dataset.test_mask], dataset.labels[dataset.test_mask])

    return test_accuracy, test_loss


def accuracy(logits, labels, mask):
    pred = logits[mask].argmax(dim=1)
    correct_pred_number = torch.eq(pred, labels[mask]).sum().item()
    acc = correct_pred_number / mask.sum().item()
    return acc


best_val_loss = math.inf
updated_best_loss_on_epoch = 0
for epoch in range(epochs):
    train_loss = train()
    train_accuracy, validation_accuracy, validation_loss = evaluate()
    print(f"Epoch {epoch + 1}/{epochs} , Train Loss {train_loss:.4f} , Train Accuracy {train_accuracy:.4f} , Validation Loss {validation_loss:.4f} , Validation Accuracy {validation_accuracy:.4f}")
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        updated_best_loss_on_epoch = epoch
        torch.save(net, model_file)
    if abs(epoch - updated_best_loss_on_epoch) >= 10:
        print(f"Early stopping on Validation loss on epoch {epoch+1}")
        break

test_accuracy, test_loss = test()
print(f"Test Loss {test_loss:.4f} , Test Accuracy {test_accuracy:.4f}")
