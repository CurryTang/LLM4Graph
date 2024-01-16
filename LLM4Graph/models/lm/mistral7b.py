import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    

def get_detailed_instruct(task_description: str, query: str) -> str:
    if task_description == '':
        return query
    else:
        return f'Instruct: {task_description}\nQuery: {query}'


def get_mistral_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')
    return tokenizer, model


def get_mistral_embedding(queries, task_descriptions):
    max_length = 4096
    tokenizer, model = get_mistral_embedding_model()
    for q, t in zip(queries, task_descriptions):
        input_texts = [get_detailed_instruct(t, q)]
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings  