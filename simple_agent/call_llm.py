import hashlib
import json
import os
from typing import Callable,Dict
import requests
from time import time

# Replace with environment variable
llm_cache_file_base='llm_cache.json'

def _load_cache(cache_file:str)->Dict:
    cache = {}
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as j_file:
                cache = json.load(j_file)
    except JSONDecodeError as jse:
        print(f'JSONDecodeError for {cache_file} - {jse}')
    finally:
        return cache

def _make_hash_prompt(prompt):
    # Create a SHA-256 hash of the prompt
    return hashlib.sha256(prompt.encode()).hexdigest()

def _record_cache(cache, cache_file):
    with open(cache_file, 'w') as j_file:
        json.dump(cache, j_file)

def create_call_llm_for_ip(ip_address_and_port: str) -> Callable[[str], str]:
    print(f"call_llm for {ip_address_and_port} made")
    def call_llm(new_prompt:str) -> str:
        #print(f"{new_prompt}\n\n\n")
        hashed_prompt=_make_hash_prompt(new_prompt)
        llm_cache_file=os.path.join("cache",llm_cache_file_base)
        cache_resp=_load_cache(llm_cache_file)
        msg=""
        if hashed_prompt in cache_resp:
            #print("Hash found!")
            msg=cache_resp[hashed_prompt]
        else:
            
            print(f"Calling LLM api on {ip_address_and_port}")
            """
            with open("llm_input.txt",'a') as file:
                file.write(f"prompt:\n{new_prompt}\n\n")
            """
            start=time()
            resp_msg=requests.post(f"{ip_address_and_port}/call_llm", json={'prompt':new_prompt})
            end=time()
            #print(f"LLM call took: {end-start}")
            resp=resp_msg.json()
            """
            with open("llm_output.txt",'a') as file:
                file.write(f"response:\n{resp['message']}\n\n")
            """
            msg=resp['message']
            cache_resp[hashed_prompt]=msg
            _record_cache(cache_resp, llm_cache_file)
        return msg
    return call_llm
