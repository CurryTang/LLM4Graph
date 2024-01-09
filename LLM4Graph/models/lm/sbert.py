import torch
from sentence_transformers import SentenceTransformer


def sbert(device):
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/tmp', device=device).to(device)
    return model 

def get_sbert_embedding(texts):
    sbert_model = sbert('cuda')
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)