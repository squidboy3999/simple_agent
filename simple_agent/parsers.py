""" Pure functions used for parsing a response from a LLM """
import re
from typing import List, Dict
from enum import Enum

class StringExtractType(Enum):
    KV_PAIR = 1
    BLOCK = 2
    ITEM_LIST =3

def _kv_extract_helper(response:str,labels:list[str])->Dict:
    result={}
    # Extract key-value pairs that match the format 'key: value'
    kv_pattern = r'(\w[\w\s]*?):\s*(.+)'
    matches = re.findall(kv_pattern, response)
    for match in matches:
        key, value = match
        if key.strip() in labels:
            result[key.strip()] = value.strip()
    return result

def _block_extract_helper(response: str, labels: List[str]) -> Dict[str, str]:
    result = {}
    for label in labels:
        # Primary attempt: Extract block of text between START <label>: and END <label>
        block_pattern = rf'{label}:\s*(.*?)(?:\n\s*END {label}|\Z)'
        match = re.search(block_pattern, response, re.DOTALL)
        if match:
            text = match.group(1).strip().replace('\nEND','')
            # Remove substrings that start with '<' and end with '>'
            text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
            if len(text)>0:
                result[label] = text
        else:
            # Fallback: If label is not found, extract the first block of text (until END or next label)
            fallback_pattern = rf'\s*(.*?)(?:\n\s*END|\Z)'
            fallback_match = re.search(fallback_pattern, response, re.DOTALL)
            if fallback_match:
                text = fallback_match.group(1).strip().replace('\nEND','')
                # Remove substrings that start with '<' and end with '>'
                text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
                if len(text)>0:
                    result[label] = text
    return result

def _item_list_extract_helper(response: str, labels: List[str]) -> Dict[str, List[str]]:
    result = {}
    for label in labels:
        # Primary attempt: Extract list in [item1, item2, ...] format
        list_pattern = rf'{label}:\s*\[(.*?)\]'
        match = re.search(list_pattern, response)
        if match:
            items = [item.strip() for item in match.group(1).split(',')]
            result[label] = items
        else:
            # Secondary attempt: Extract list formatted with dashes on new lines
            block_pattern = rf'{label}:\s*\n((?:\s*-\s*.+\n?)+)'
            match = re.search(block_pattern, response)
            if match:
                items = [item.strip()[2:] for item in match.group(1).strip().splitlines()]
                result[label] = items
            else:
                # Fallback: Extract any loose list formatted with dashes without the label
                fallback_pattern = r'((?:\s*-\s*.+\n?)+)'
                fallback_match = re.search(fallback_pattern, response)
                if fallback_match:
                    items = [item.strip()[2:] for item in fallback_match.group(0).strip().splitlines()]
                    result[label] = items
    return result

def _string_extract_to_dict(response:str,extract_type:StringExtractType,labels:list[str])->Dict:
    result = {}

    if extract_type == StringExtractType.KV_PAIR:
        result=_kv_extract_helper(response,labels)

    elif extract_type == StringExtractType.BLOCK:
        result=_block_extract_helper(response,labels)

    elif extract_type == StringExtractType.ITEM_LIST:
        result=_item_list_extract_helper(response,labels)

    return result

def _basic_parser(response:str,type:StringExtractType,labels:str)->Dict:
    extract_dict=_string_extract_to_dict(response,type,labels)
    if extract_dict=={}:
        extract_dict['error']="string extraction error"
    return extract_dict

def kv_parser(response:str,labels:str)->Dict:
    return _basic_parser(response,StringExtractType.KV_PAIR,labels)

def block_parser(response:str,labels:str)->Dict:
    return _basic_parser(response,StringExtractType.BLOCK,labels)

def item_list_parser(response:str,labels:str)->Dict:
    return _basic_parser(response,StringExtractType.ITEM_LIST,labels)