"""
    An example of using LLMs as predictors to solve various tasks.
    * Normal Node classification
    * OOD Node classification
"""
import sys 
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from LLM4Graph.data.PlanetoidTAG import TAGPlanetoidDataset
from LLM4Graph.data.OGBNodeTAG import TAGOGBNodeDataset
from LLM4Graph.data.WikiCSTAG import TAGWikiCSDataset
from LLM4Graph.data.prompts import *
from LLM4Graph.utils.misc import generate_timestamp_filename
from LLM4Graph.utils.llm.api import efficient_openai_text_api
import torch

## Normal Node classification
datasets = ['cora', 'citeseer', 'pubmed', 'arxiv', 'products', 'wikics']

for d in datasets:
    if d in ['cora', 'citeseer', 'pubmed']:
        dataset = TAGPlanetoidDataset(root = './data', name = d, split = 'semi')
    elif d == 'wikics':
        dataset = TAGWikiCSDataset(root = './data', name = d, split = 'default')
    else:
        dataset = TAGOGBNodeDataset(root = './data', name = d, split = 'default')

    seeds = [0, 1, 2]
    accs = []
    budget = 100
    for seed in seeds:
        selected_indices = torch.arange(len(dataset))[dataset.test_masks[seed]]
        selected_indices = torch.randperm(selected_indices)[:budget]

        raw_texts = dataset.get_raw_texts()
        data = dataset[0]




