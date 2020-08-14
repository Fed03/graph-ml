from typing import List
from graphml.ModelRunner import EpochStat
import os
import csv

from graphml.metrics import Metric


def write_train_epochs_stats(dir:str ,stats: List[EpochStat]) -> None:
    with open(os.path.join(dir, "epochs_stats.csv"), "w", newline="") as csv_file:
        train_stats_dict = map(lambda x: x.asdict(), stats)
        train_stats_dict = map(
            lambda x: {**x, "epoch": x["epoch"]+1}, train_stats_dict)
        train_stats_dict = list(map(
            lambda x: {k: v for k, v in x.items() if k != 'total_epochs'}, train_stats_dict))
        writer = csv.DictWriter(csv_file, train_stats_dict[0].keys())
        writer.writeheader()
        writer.writerows(train_stats_dict)

def write_test_results(dir: str, run_id: str, results: List[Metric]):
    row = {"run_id": run_id}
    row.update({m.name: m.value for m in results})

    filename = os.path.join(dir, "tests_stats.csv")
    
    file_exists = os.path.exists(filename)
    file_flag = "a" if file_exists else "w"

    with open(filename,file_flag,newline="") as csv_file:
        writer = csv.DictWriter(csv_file, row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
