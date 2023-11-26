import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
import gdown
import pandas as pd
from LLM4Graph.utils.data_utils import get_mask
from torch_geometric.datasets import WikiCS

class TAGWikiCSDataset(InMemoryDataset):
    """
        The TAG version of the WikiCS dataset.
    """
    url_lib = {
        "wikics_sbert.pt": "https://drive.google.com/file/d/1hDbk5gbWPRfPcQFtsQasTwkI45aauNON",
        "wikics_raw.parquet":  "https://drive.google.com/file/d/1QB8CG7H5EPV-yzth_HpBhjCdfl9xtPFK"
    }
    def __init__(self, root = None, name = "wikics", split = "random",
                 train_ratio = 0.6, val_ratio = 0.2, test_ratio = 0.2, num_splits = 10,
                  transform = None, pre_transform = None, pre_filter = None, force_reload = False):
        self.name = name.lower()
        self.split = split.lower()
        self.seeds = [i for i in range(num_splits)]
        self.train_ratio = train_ratio 
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        assert self.name  == 'wikics'
        assert self.split in ["random", "default"]

        super().__init__(root, transform, pre_transform, pre_filter, force_reload)

        data = torch.load(self.processed_paths[1])
        self.data, self.slices = self.collate(
            [data]
        )

        self.raw_texts = pd.read_parquet(self.processed_paths[0])


    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed", self.split)
    
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
            gdown.download(
                self.url_lib[name], osp.join(self.raw_dir, name), quiet = True, fuzzy = True
            )
    def get_raw_texts(self):
        return self.raw_texts

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
                if self.split == "default":
                    data_file = WikiCS("/tmp")[0]
                    train_masks = data_file.train_mask
                    val_masks = data_file.val_mask
                    test_masks = data_file.test_mask
                    assert len(self.seeds) < data_file.train_mask.shape[1], "Default split only supports 20 seeds."
                    train_masks = [train_masks[:, i] for i in range(len(self.seeds))]
                    val_masks = [val_masks[:, i] for i in range(len(self.seeds))]
                    test_masks = [test_masks[:, i] for i in range(len(self.seeds))]
                else:
                    train_masks, val_masks, test_masks = get_mask(data, self.split, self.train_ratio, self.val_ratio, self.test_ratio, self.seeds)
                data = data if self.pre_transform is None else \
                    self.pre_transform(data)

                data.train_masks = train_masks
                data.val_masks = val_masks 
                data.test_masks = test_masks
                torch.save(data, process_file)

            