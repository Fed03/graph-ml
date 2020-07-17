import os
from typing import Callable
from graphml.datasets.InternalData import InternalData


class DatasetLoader():
    def __init__(self, dataset_name: str, base_path: str, *transform: Callable[[InternalData], InternalData]):
        self._dataset_name = dataset_name
        self._root_path = os.path.join(base_path, "data", self._dataset_name)
        self._transform_funcs = transform

    @property
    def _pretty_name(self):
        return self._dataset_name.capitalize()

    @property
    def _raw_folder(self):
        return os.path.join(self._root_path, "raw")

    @property
    def _processed_file_path(self):
        return os.path.join(self._root_path, f"{self._dataset_name}.processed.pt")

    def load(self):
        self._download_dataset()
        self._process()
        return self._internal_data


class PPIDataset(DatasetLoader):
    url = "https://data.dgl.ai/dataset/ppi.zip"
    
    def __init__(base_path: str, *transform: Callable[[InternalData], InternalData]):
        super().__init__("ppi",base_path,*transform)

    def _download_dataset(self):
        if os.path.exists(self._raw_folder):
            print(f"The {self._pretty_name} dataset is already downloaded.")
        else:
            os.makedirs(self._raw_folder)

            print(f"Downloading {self._pretty_name} dataset files...")
            for file_name in tqdm(self._raw_file_names):

                response = requests.get(f"{self.url}/{file_name}")
                with open(os.path.join(self._raw_folder, file_name), "wb") as target:
                    target.write(response.content)
            print("Download completed.")
