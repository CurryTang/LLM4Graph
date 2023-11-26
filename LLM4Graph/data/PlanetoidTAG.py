import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
import gdown
import pandas as pd
from LLM4Graph.utils.data_utils import get_mask

class TAGPlanetoidDataset(InMemoryDataset):
    """
        The TAG version of the classical Planetoid dataset.
    """
    url_lib = {
        "cora_sbert.pt": "https://drive.google.com/file/d/1e72LU6DTcAJ7o8gpE36QhuNpLfsRJUpM",
        "cora_raw.parquet":  "https://drive.google.com/file/d/1wD1lHj3Tk0gzOwPuQJk54CHni_mtuIJG",
        "citeseer_sbert.pt": "https://drive.google.com/file/d/10YjtJ_4Yau-lgx6qzobW27H6199BZ3tf",
        "citeseer_raw.parquet": "https://drive.google.com/file/d/1u8HxqhIja_h_Ny8TMudHBVBD-TwE6tqU", 
        "pubmed_sbert.pt": "https://drive.google.com/file/d/13XWoAs667QREkx3OqDpbxSUEPAqY8zVj", 
        "pubmed_raw.parquet": "https://drive.google.com/file/d/1Xwmpx5HQh82fN8_46WFuQRNJt7jT5909"
    }
    def __init__(self, root = None, name = "Cora", split = "random",
                 train_ratio = 0.6, val_ratio = 0.2, test_ratio = 0.2, num_splits = 10,
                  transform = None, pre_transform = None, pre_filter = None, force_reload = False, 
                  max_df = 1., min_df = 1, max_features = None, homo_split = 0.5):
        self.name = name.lower()
        self.split = split.lower()
        self.seeds = [i for i in range(num_splits)]
        self.max_df = max_df 
        self.min_df = min_df 
        self.max_features = max_features
        self.homo_split = homo_split
        self.train_ratio = train_ratio 
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        assert self.name in ["cora", "citeseer", "pubmed"]
        assert self.split in ["semi", "high", "random", "ood_degree", 
                              "ood_homo", "ood_concept"]

        super().__init__(root, transform, pre_transform, pre_filter, force_reload)

        data = torch.load(self.processed_paths[1])
        self.data, self.slices = self.collate(
            [data]
        )


    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")
    
    @property
    def raw_file_names(self) -> str:
        return [
            "{}_raw.parquet".format(self.name),
            "{}_sbert.pt".format(self.name)
        ]
    
    @property
    def processed_file_names(self) -> str:
        return [
            "{}_raw.parquet".format(self.name),
            "{}_sbert.pt".format(self.name)
        ]
    
    def download(self):
        for name in self.raw_file_names:
            gdown(
                self.url_lib[name], osp.join(self.raw_dir, name, quiet = True)
            )
    
    def __repr__(self) -> str:
        return f"{self.name}TAG()"

    def process(self):
        for name in self.raw_file_names:
            raw_file = osp.join(self.raw_dir, name)
            process_file = osp.join(self.processed_dir, name)

            if "raw" in name:
                raw_file_df = pd.read_parquet(raw_file)
                raw_file_df.to_parquet(process_file, compression = "snappy")
            else:
                data = torch.load(raw_file)
                text_data = pd.read_parquet(self.processed_paths[0])
                train_masks, val_masks, test_masks = get_mask(
                    data, self.split, self.train_ratio, self.val_ratio, self.test_ratio, self.seeds, 
                    text_data, self.homo_split, self.max_df, 
                    self.min_df, self.max_features
                )

                data = data if self.pre_transform is None else \
                    self.pre_transform(data)

                data.train_masks = train_masks
                data.val_masks = val_masks 
                data.test_masks = test_masks
                torch.save(data, process_file)

            