"""
    An example of using LLMs as predictors to solve various tasks.
    * Normal Node classification
    * OOD Node classification
"""
import sys 
import os
## only for test, no need when released as package
sys.path.append(os.path.join(os.getcwd(), '..'))
from LLM4Graph.data.PlanetoidTAG import TAGPlanetoidDataset
from LLM4Graph.data.OGBNodeTAG import TAGOGBNodeDataset
from LLM4Graph.data.WikiCSTAG import TAGWikiCSDataset
from LLM4Graph.data.prompts.prompts import TemplatePrompt
from LLM4Graph.utils.misc import get_first_k_words
from LLM4Graph.utils.llm.api import efficient_openai_text_api
import torch

prompts = {
    "cora": {
        "background": "You are gonna classify a paper",
        "instruction": "Please classify the paper into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "citeseer": {
        "background": "You are gonna classify a paper",
        "instruction": "Please classify the paper into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "pubmed": {
        "background": "You are gonna classify a paper",
        "instruction": "Please classify the paper into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "arxiv": {
        "background": "You are gonna classify a paper",
        "instruction": "Please classify the paper into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "products": {
        "background": "You are gonna classify a product",
        "instruction": "Please classify the product into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "wikics": {
        "background": "You are gonna classify a Wikipedia article",
        "instruction": "Please classify the article into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    }
}


## Normal Node classification
datasets = ['cora', 'citeseer', 'pubmed', 'arxiv', 'products']

for d in datasets:
    if d in ['cora', 'citeseer', 'pubmed']:
        dataset = TAGPlanetoidDataset(root = './data', name = d, split = 'semi')
    else:
        dataset = TAGOGBNodeDataset(root = './data', name = d, split = 'default')

    seeds = [0, 2]
    accs = []
    budget = 200
    cutoff = 256
    demos = 3
    for seed in seeds:
        selected_indices = torch.arange(len(dataset))[dataset.test_masks[seed]]
        selected_indices = torch.randperm(selected_indices)[:budget]

        train_indices = torch.arange(len(dataset))[dataset.train_masks[seed]]
        demo_indices = torch.randperm(train_indices)[:demos]

        raw_texts = dataset.get_raw_texts()
        data = dataset[0]

        # Replace NaN with an empty string
        raw_texts.fillna('', inplace=True)

        # Concatenate 'title' and 'content' with a space, handling NaN values
        clean_raw_texts = raw_texts.apply(lambda row: get_first_k_words(row['title'] + ' ' + row['content'], k = cutoff), axis=1).tolist()
        
        demos = [
            clean_raw_texts[i.item()] for i in demo_indices
        ]

        
        
        ## zero shot
        zero_shot_prompt = [TemplatePrompt(
            t,
            background=prompts[d]['background'],
            instruction=prompts[d]['instruction'],
            question=prompts[d]['question']
        ) for t in clean_raw_texts]







