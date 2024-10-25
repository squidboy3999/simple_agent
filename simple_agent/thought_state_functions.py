from typing import Callable, Dict, Tuple, List
from action_functions import ActionFunctions
from parsers import kv_parser, block_parser, item_list_parser
from simple_agent.summarize_prompts import *
from simple_agent.debator_prompts import *
from enum import Enum, auto

class ThoughtState(Enum):
    SUMMARIZE_AS_BULLETS = auto()
    FILTER_FOR_BIAS = auto()
    REWRITE_FOR_BIAS = auto()
    BIAS_1_VIEWPOINT = auto()
    BIAS_2_VIEWPOINT = auto()
    BIAS_1_CROSS_EXAMINATION = auto()
    BIAS_2_CROSS_EXAMINATION = auto()
    BIAS_1_CROSS_EXAMINATION_ANSWERS = auto()
    BIAS_2_CROSS_EXAMINATION_ANSWERS = auto()
    BIAS_1_DIFF_VIEWPOINT_SUMMARY=auto()
    BIAS_2_DIFF_VIEWPOINT_SUMMARY=auto()
    BIAS_1_CONCLUSION = auto()
    BIAS_2_CONCLUSION = auto()
    STORE_INFO = auto()
    END = auto()


def _update_kv_memory_helper(new_dict:Dict,
                             memory:Dict,
                             key_list:List[str],
                             next_thought_state:ThoughtState,
                             filter_values:List[str],
                             response:str) -> List[Tuple[ThoughtState, Dict]]:
    for key in key_list:
        try:
            if new_dict[key] in filter_values:
                memory["stopped"]=f"-- {new_dict[key]} -- found in respone"
                return [(ThoughtState.STORE_INFO,memory)]
            else:
                memory[key]=new_dict[key]
        except KeyError as key_e:
            memory["response"]=response
            memory["error"]=f"KeyError - {key_e}"
            return [(ThoughtState.STORE_INFO,memory)]
    return [(next_thought_state,memory)]

def _update_list_memory_helper(key_list_dict:Dict,memory:Dict,key_list:List[str],next_thought_state:ThoughtState,response:str) -> List[Tuple[ThoughtState, Dict]]:
    next_action_tuples=[]
    for key in key_list:
        try:
            for key_item in key_list_dict[key]:
                new_memory=memory.copy()
                new_memory[key[:-1] if key.endswith('s') else key]=key_item
                next_action_tuples.append((next_thought_state,new_memory))
        except KeyError as key_e:
            new_memory=memory.copy()
            new_memory["response"]=response
            new_memory["error"]=f"KeyError - {key_e}"
            next_action_tuples.append((ThoughtState.STORE_INFO,new_memory))
    return next_action_tuples

# Function placeholders for each state
def summarize_as_bullets(memory: Dict, action_functions: ActionFunctions, next_state:ThoughtState) -> List[Tuple[ThoughtState, Dict]]:
    """ takes raw text and splits the memory so each bullet can be its own chain """
    prompt,key_list=get_summarize_as_bullets_prompt_and_labels(memory)
    response=action_functions.call_llm(prompt)
    action_functions.logger(f"function - summarize_as_bullets:\n\nprompt - \n{prompt}\n\n\nresponse - \n{response}\n\n\n\n\n")
    key_list_dict=item_list_parser(response,key_list)
    return _update_list_memory_helper(key_list_dict,memory,key_list,next_state,response)


def filter_for_bias(memory: Dict, action_functions: ActionFunctions, next_state:ThoughtState) -> List[Tuple[ThoughtState, Dict]]:
    """ discards irrelevant bullets for that bias """
    prompt,key_list=get_rate_knowledge_by_bias_prompt_and_labels(memory)
    response=action_functions.call_llm(prompt)
    action_functions.logger(f"function - filter_for_bias:\n\nprompt - \n{prompt}\n\n\nresponse - \n{response}\n\n\n\n\n")
    new_dict=kv_parser(response,key_list)
    return _update_kv_memory_helper(new_dict,memory,key_list,next_state,['not_relevant','not relevant'],response)


def rewrite_for_bias(memory: Dict, action_functions: ActionFunctions, next_state:ThoughtState) -> List[Tuple[ThoughtState, Dict]]:
    """ Rewrites a bullet so it is more targeted for a bias """
    prompt,key_list=get_rewrite_knowledge_by_bias_prompt_and_labels(memory)
    response=action_functions.call_llm(prompt)
    action_functions.logger(f"function - rewrite_for_bias:\n\nprompt - \n{prompt}\n\n\nresponse - \n{response}\n\n\n\n\n")
    new_dict=kv_parser(response,key_list)
    return _update_kv_memory_helper(new_dict,memory,key_list,next_state,[],response)

def debate_bias_bias_viewpoint(memory:Dict, 
                               action_functions:ActionFunctions,
                               next_state:ThoughtState,
                               bias_key:str)->List[Tuple[ThoughtState, Dict]]:
    prompt,key_list=get_debate_bias_viewpoint_prompt_and_labels(memory,bias_key)
    response=action_functions.call_llm(prompt)
    action_functions.logger(f"function - debate_bias_bias_viewpoint:\n\nprompt - \n{prompt}\n\n\nresponse - \n{response}\n\n\n\n\n")
    new_dict=block_parser(response,key_list)
    return _update_kv_memory_helper(new_dict,memory,key_list,next_state,[],response)

def debate_bias_cross_examination(memory:Dict, 
                                  action_functions:ActionFunctions,
                                  next_state:ThoughtState,
                                  bias_key_1:str,
                                  bias_key_2:str)->List[Tuple[ThoughtState, Dict]]:
    prompt,key_list=get_debate_bias_cross_examination_prompt_and_labels(memory,bias_key_1,bias_key_2)
    response=action_functions.call_llm(prompt)
    action_functions.logger(f"function - debate_bias_cross_examination:\n\nprompt - \n{prompt}\n\n\nresponse - \n{response}\n\n\n\n\n")
    new_dict=block_parser(response,key_list)
    return _update_kv_memory_helper(new_dict,memory,key_list,next_state,[],response)

def debate_bias_cross_examination_answers(memory:Dict, 
                                          action_functions:ActionFunctions,
                                          next_state:ThoughtState,
                                          bias_key_1:str,
                                          bias_key_2:str)->List[Tuple[ThoughtState, Dict]]:
    prompt,key_list=get_debate_bias_cross_examination_answers_prompt_and_labels(memory,bias_key_1,bias_key_2)
    response=action_functions.call_llm(prompt)
    action_functions.logger(f"function - debate_bias_cross_examination_answers:\n\nprompt - \n{prompt}\n\n\nresponse - \n{response}\n\n\n\n\n")
    new_dict=block_parser(response,key_list)
    return _update_kv_memory_helper(new_dict,memory,key_list,next_state,[],response)

def debate_bias_diff_viewpoint_summary(memory:Dict, 
                                       action_functions:ActionFunctions,
                                       next_state:ThoughtState,
                                       bias_key_1:str,
                                       bias_key_2:str)->List[Tuple[ThoughtState, Dict]]:
    prompt,key_list=get_debate_bias_diff_viewpoint_summary_prompt_and_labels(memory,bias_key_1,bias_key_2)
    response=action_functions.call_llm(prompt)
    action_functions.logger(f"function - debate_bias_diff_viewpoint_summary:\n\nprompt - \n{prompt}\n\n\nresponse - \n{response}\n\n\n\n\n")
    new_dict=block_parser(response,key_list)
    return _update_kv_memory_helper(new_dict,memory,key_list,next_state,[],response)

def debate_bias_conclusion(memory:Dict, 
                           action_functions:ActionFunctions,
                           next_state:ThoughtState,
                           bias_key_1:str,
                           bias_key_2:str)->List[Tuple[ThoughtState, Dict]]:
    prompt,key_list=get_debate_bias_conclusion_prompt_and_labels(memory,bias_key_1,bias_key_2)
    response=action_functions.call_llm(prompt)
    action_functions.logger(f"function - debate_bias_conclusion:\n\nprompt - \n{prompt}\n\n\nresponse - \n{response}\n\n\n\n\n")
    new_dict=block_parser(response,key_list)
    return _update_kv_memory_helper(new_dict,memory,key_list,next_state,[],response)

def store_info(memory: Dict,action_functions: ActionFunctions, next_state:ThoughtState) -> List[Tuple[ThoughtState, Dict]]:
    tmp_str=""
    for key,value in memory.items():
        tmp_str+=f"{key}: {value}\n"
    action_functions.store_info(f"{tmp_str}\n")
    return [(next_state,{})]