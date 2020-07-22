import os
import torch
import requests
from tqdm import tqdm
from .InternalData import GraphData
from typing import Any, Callable, List, Union


class BaseDatasetLoader():
    def __init__(self, dataset_name: str, base_path: str, *transform: Callable[[GraphData], GraphData]):
        self._dataset_name = dataset_name
        self._root_path = os.path.join(base_path, "data", self._dataset_name)
        self._transform_funcs = transform

    @property
    def _pretty_name(self) -> str:
        return self._dataset_name.capitalize()

    @property
    def _url(self) -> str:
        raise NotImplementedError

    @property
    def _raw_file_names(self) -> Union[List[str], str]:
        raise NotImplementedError

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self._root_path, "raw")

    @property
    def _processed_file_path(self) -> str:
        transform_prefix = ""
        if self._transform_funcs:
            transforms_name = map(
                lambda t: t.__class__.__name__, self._transform_funcs)
            transform_prefix = "_".join(sorted(transforms_name)) + "."
        return os.path.join(self._root_path, f"{self._dataset_name}.{transform_prefix.lower()}processed.pt")

    def load(self) -> Any:
        self._download_dataset()
        self._process()
        return self._internal_data

    def _download_dataset(self):
        if os.path.exists(self._raw_folder):
            print(f"The {self._pretty_name} dataset is already downloaded.")
        else:
            os.makedirs(self._raw_folder)

            print(f"Downloading {self._pretty_name} dataset files...")
            if isinstance(self._raw_file_names, list):
                for file_name in tqdm(self._raw_file_names):
                    self._dowload_file(f"{self._url}/{file_name}", file_name)
            else:
                self._dowload_file(
                    f"{self._url}/{self._raw_file_names}", self._raw_file_names)

            print("Download completed.")

    def _dowload_file(self, url: str, file_name: str):
        response = requests.get(url)
        with open(os.path.join(self._raw_folder, file_name), "wb") as target:
            target.write(response.content)

    def _process(self):
        if not os.path.exists(self._processed_file_path):
            print("Processing raw dataset files...")
            self._internal_data = self._process_raw_files()
            torch.save(self._internal_data, self._processed_file_path)
        else:
            self._internal_data = torch.load(self._processed_file_path)

        print(f"{self._pretty_name} dataset correctly loaded.")

    def _process_raw_files(self) -> Any:
        raise NotImplementedError

    def _apply_transforms(self, data: GraphData) -> GraphData:
        for transform in self._transform_funcs:
            data = transform(data)

        return data
