import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import os.path as osp
import gdown
import pandas as pd
from LLM4Graph.utils.data_utils import get_mask
import os


class TAGOGBNodeDataset(InMemoryDataset):
    """
        The TAG version of the OGB dataset.
        For Arxiv, we support OOD split and prediction file from TAPE.
    """
    url_lib = {
        "arxiv_sbert.pt": "https://drive.google.com/file/d/17-9GzEWcrOkx4KuMTG5GykwaxSICOKoX",
        "arxiv_raw.parquet":  "https://drive.google.com/file/d/1T0bNq4V0l04b7M6NOCDuGcdcKV3rfum_",
        "products_sbert.pt": "https://drive.google.com/file/d/11T2FBC-mKDpqy7ULFrhF7tU5Nc8lMgzt",
        "products_raw.parquet": "https://drive.google.com/file/d/1eTqDq1kSrbuQXC1pNOV7OhLemNTY_MNg"
    }

    tape_pred = "https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/gpt_preds/ogbn-arxiv.csv"
    data_map = {
        "arxiv": "ogbn-arxiv",
        "products": "ogbn-products"
    }
    mask_link = {
        "arxiv": "https://drive.google.com/file/d/1iwEqcLVXmsYQWdYrqDdZq9aP0SN4OAAf",
        "products": "https://drive.google.com/file/d/1A0n1QqvtMoT_ec0txGAzxyGT7_Zy2Mea"
    }
    ood_url = "'https://drive.google.com/file/d/1OyMOwT4bn_4fLdpl5B3ie18OmGsUNQxS/view?usp=sharing'"
    def __init__(self, root = None, name = "Arxiv", split = "random",
                 train_ratio = 0.6, val_ratio = 0.2, test_ratio = 0.2, num_splits = 10,
                  transform = None, pre_transform = None, pre_filter = None, force_reload = False, 
                  homo_split = 0.5):
        self.name = name.lower()
        self.split = split.lower()
        self.seeds = [i for i in range(num_splits)]
        self.homo_split = homo_split
        self.train_ratio = train_ratio 
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        assert self.name in ["cora", "citeseer", "pubmed"]
        assert (self.split in ["default", "random", "ood_degree", 
                              "ood_homo", "ood_time"] and self.name == 'arxiv') \
                              or (self.split in ['default', 'random'] and self.name == 'products')

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
            "{}_sbert.pt".format(self.name),
            "{}_default_masks".format(self.name)
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

                if self.split == "default":
                    mask_file = osp.join(self.raw_dir, "{}_default_masks".format(self.name))
                    train_mask, val_mask, test_mask = mask_file
                    train_masks = [train_mask]
                    val_masks = [val_mask]
                    test_masks = [test_mask]
                elif self.split == 'ood_time' or self.split == 'ood_degree':
                    path = gdown.download(self.ood_url,
                                          output= osp.join(self.raw_dir, self.name + '.zip'),
                                          quiet = True)
                    extract_zip(path, self.raw_dir)
                    os.unlink(path)
                else:
                    train_masks, val_masks, test_masks = get_mask(
                        data, self.split, self.train_ratio, self.val_ratio, self.test_ratio, self.seeds, 
                        homo_split=self.homo_split
                    )

                data = data if self.pre_transform is None else \
                    self.pre_transform(data)

                data.train_masks = train_masks
                data.val_masks = val_masks 
                data.test_masks = test_masks
                torch.save(data, process_file)

            