""" Pure functions used for building prompts for LLM """
from typing import List

def _make_list_str(items:List[str],indent:int) -> str:
    indent_str = ' ' * indent
    new_str=""
    for item in items:
        new_str+=f"{indent_str}- {item}\n"
    return new_str

def _make_dict_str(items:dict,indent:int) -> str:
    indent_str = ' ' * indent
    new_str=""
    for k,v in items.items():
        new_str+=f"{indent_str}{k}: {v}\n"
    return new_str

def _make_example(example_input:str,leading_token:str,example_string:str) -> str:
    return example_input+"\n"+leading_token+"\n"+example_string+"\n"

def make_list_example(example_input:str,leading_token:str,example_items:List[str]) -> str:
    return _make_example(example_input, leading_token, _make_list_str(example_items,2))
    
def make_dict_example(example_input:str,leading_token:str,example_items:dict) -> str:
    return _make_example(example_input, leading_token, _make_dict_str(example_items,2))

def make_prompt_template(description:str,examples:List[str],resp_template:str,input_keys:List[str],leading_token:str)->str:
    str_examples=""
    inputs=""
    cnt=1
    for example in examples:
        str_examples+=f"example {cnt}\n{example}\n"
        cnt=cnt+1
    for input_key in input_keys:
        inputs+=f"|{input_key}|\n"
    ex_part="Consider these examples - "+str_examples+"\n"
    tmp_part="follow this template:\n"+resp_template+"\n"
    in_part="Use the following input\n"+inputs+"\n"+leading_token+"\n"
    return description+"\n"+ex_part+tmp_part+in_part

def prompt_from_template(prompt_template:str,state:dict)->str:
    prompt=prompt_template
    for k,v in state.items():
        prompt=prompt.replace(f"|{k}|",v)
    return prompt