import os.path as osp
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def average_pool(last_hidden_states,
                 attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def batched_data(inputs, batch_size):
    return [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]

def get_e5_large_embedding(texts, device, dataset_name = 'cora', batch_size = 64, cache_out = '/tmp', update = True):
    output_path = osp.join(cache_out, dataset_name + "_e5_embedding.pt")
    if osp.exists(output_path) and not update:
        return torch.load(output_path, map_location='cpu')
    texts = ["query: " + x for x in texts]
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2', cache_dir='/tmp')
    model = AutoModel.from_pretrained('intfloat/e5-large-v2', cache_dir='/tmp').to(device)
    # Tokenize the input texts
    output = []
    with torch.no_grad():
        for batch in tqdm(batched_data(texts, batch_size)):
            batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            output.append(embeddings.cpu())
            del batch_dict
    output = torch.cat(output, dim = 0)
    torch.save(output, output_path)
    return output