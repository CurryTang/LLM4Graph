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
        "arxiv_sbert.pt": "https://drive.google.com/uc?id=17-9GzEWcrOkx4KuMTG5GykwaxSICOKoX",
        "arxiv_raw.parquet":  "https://drive.google.com/uc?id=1T0bNq4V0l04b7M6NOCDuGcdcKV3rfum_",
        "products_sbert.pt": "https://drive.google.com/uc?id=11T2FBC-mKDpqy7ULFrhF7tU5Nc8lMgzt",
        "products_raw.parquet": "https://drive.google.com/uc?id=1eTqDq1kSrbuQXC1pNOV7OhLemNTY_MNg",
        "arxiv_default_masks": "https://drive.google.com/uc?id=1iwEqcLVXmsYQWdYrqDdZq9aP0SN4OAAf",
        "products_default_masks": "https://drive.google.com/uc?id=1A0n1QqvtMoT_ec0txGAzxyGT7_Zy2Mea"
    }

    tape_pred = "https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/gpt_preds/ogbn-arxiv.csv"
    data_map = {
        "arxiv": "ogbn-arxiv",
        "products": "ogbn-products"
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
        assert self.name in ["arxiv", "products"]
        assert (self.split in ["default", "random", "ood_degree", 
                              "ood_homo", "ood_time"] and self.name == 'arxiv') \
                              or (self.split in ['default', 'random'] and self.name == 'products')

        super().__init__(root, transform, pre_transform, pre_filter, force_reload)

        data = torch.load(self.processed_paths[1])
        self.data, self.slices = self.collate(
            [data]
        )

        self.raw_texts = pd.read_parquet(self.processed_paths[0])
        self.tape = torch.load(self.processed_paths[2])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed", self.split)
    
    @property
    def raw_file_names(self) -> str:
        if self.name == 'arxiv':
            return [
                "{}_raw.parquet".format(self.name),
                "{}_sbert.pt".format(self.name),
                "{}_default_masks".format(self.name),
                "ogbn-arxiv.csv"
            ]
        else:
            return [
                "{}_raw.parquet".format(self.name),
                "{}_sbert.pt".format(self.name),
                "{}_default_masks".format(self.name)
            ]
    
    @property
    def processed_file_names(self) -> str:
        if self.name == 'arxiv':
            return [
                "{}_raw.parquet".format(self.name),
                "{}_sbert.pt".format(self.name),
                "{}_tape.pt".format(self.name)
            ]
        else:
            return [
                "{}_raw.parquet".format(self.name),
                "{}_sbert.pt".format(self.name)
            ]
    
    def download(self):
        for name in self.raw_file_names:
            if self.url_lib.get(name, None):
                gdown.download(
                    self.url_lib[name], osp.join(self.raw_dir, name), fuzzy = True
                )
    
    def get_raw_texts(self):
        return self.raw_texts
    
    def __repr__(self) -> str:
        return f"{self.name}TAG()"

    def process(self):
        for name in self.raw_file_names:
            raw_file = osp.join(self.raw_dir, name)
            process_file = osp.join(self.processed_dir, name)

            if "masks" in name:
                continue

            if "arxiv.csv" in name and self.name != 'arxiv':
                continue
            if "raw" in name:
                raw_file_df = pd.read_parquet(raw_file)
                raw_file_df.to_parquet(process_file, compression = "snappy")
            elif "arxiv.csv" in name and self.name == 'arxiv':
                download_url(self.tape_pred, self.raw_dir)
                tape_file = osp.join(self.raw_dir, "ogbn-arxiv.csv")
                first_numbers = []
                with open(tape_file, 'r') as file:
                    for line in file:
                        line = line.strip()
                        first_number = line.split(',')[0]
                        try:
                            first_numbers.append(int(first_number))
                        except:
                            first_numbers.append(1)
                tape = torch.tensor(first_numbers)
                torch.save(tape, self.processed_paths[2])
            else:
                data = torch.load(raw_file)

                if self.split == "default":
                    mask_file = torch.load(osp.join(self.raw_dir, "{}_default_masks".format(self.name)))
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
                    if self.split == 'ood_degree':
                        covariate = osp.join(path, "GOODArxiv", "degree/processed/covariate.pt")
                    elif self.split == 'ood_time':
                        covariate = osp.join(path, "GOODArxiv", "time/processed/covariate.pt")
                    train_masks = [covariate.train_mask]
                    val_masks = [covariate.val_mask]
                    test_masks = [covariate.test_mask]
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

            