import logging
from LLM4Graph.utils.llm.async_utils import process_api_requests_from_file
import asyncio 
import tiktoken
import os.path as osp
import json


async def call_async_api(request_filepath, save_filepath, request_url, api_key, max_request_per_minute, max_tokens_per_minute, sp, ss):
    await process_api_requests_from_file(
            requests_filepath=request_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_request_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name='cl100k_base',
            max_attempts=int(2),
            logging_level=int(logging.INFO),
            seconds_to_pause=sp,
            seconds_to_sleep=ss
        )
    

def generate_chat_input_file(input_text, system_prompt = "", model_name = 'gpt-3.5-turbo', temperature = 0, n = 1):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['model'] = model_name
        if system_prompt != "":
            obj['messages'] = [
                {
                    'role': 'system',
                    "content": system_prompt
                }, 
                {
                    'role': 'user',
                    'content': text
                }
            ]
        else:
            obj['messages'] = [
                {
                    'role': 'user',
                    'content': text 
                }
            ]
        obj['temperature'] = temperature
        obj['n'] = n
        jobs.append(obj)
    return jobs 


def efficient_openai_text_api(input_texts, system_prompt = "", endpoints = "https://api.openai.com/v1/chat/completions", rewrite = True, model_name = "gpt-3.5-turbo", filename = "/tmp/query.json", savepath = '/tmp/response.json', sp = 0, ss = 0, api_key = "your_key", temperature = 0, n = 1):
    if not osp.exists(savepath) or rewrite:
        jobs = generate_chat_input_file(input_texts, system_prompt, model_name=model_name, temperature = temperature, n = n)
        with open(filename, "w") as f:
            for i, job in enumerate(jobs):
                json_string = json.dumps(job)
                if job['messages'][0]['content'] != "":
                    f.write(json_string + "\n")
        asyncio.run(
            call_async_api(
                filename, save_filepath=savepath,
                request_url=endpoints,
                api_key=api_key,
                max_request_per_minute=100000000, 
                max_tokens_per_minute=9000000000,
                sp=sp,
                ss=ss
            )
        )
    openai_result = []
    with open(savepath, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            idx = json_obj[-1]
            choices = []
            if isinstance(idx, int):
                choices = [x['message']['content'] for x in json_obj[1]['choices']]
                openai_result.append((choices, idx))
            else:
                idx = json_obj[-2]
                choices = ["error"]
                openai_result.append((choices, idx))
    openai_result = sorted(openai_result, key=lambda x:x[-1])
    return openai_result
    

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == 'gpt-3.5-turbo-instruct':
        tokens_per_message = 0
        tokens_per_name = 0
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens