"""
Pure functions for transitioning state. 
All functions call pure function but can take impure functions as args.
"""
from typing import Callable, Dict, Tuple, List
from simple_agent.debator_prompts import get_debator_biases_init_memory
from itertools import product
from action_functions import ActionFunctions 
from thought_state_functions import *

# Function placeholders for each state
def _summarize_as_bullets(memory: Dict, action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return summarize_as_bullets(memory, action_functions, ThoughtState.FILTER_FOR_BIAS)

def _filter_for_bias(memory: Dict, action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return filter_for_bias(memory, action_functions, ThoughtState.REWRITE_FOR_BIAS)

def _rewrite_for_debate(memory: Dict, action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return rewrite_for_bias(memory, action_functions,ThoughtState.BIAS_1_VIEWPOINT)

def _debate_bias_1_viewpoint(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_bias_viewpoint(memory, action_functions,ThoughtState.BIAS_2_VIEWPOINT,"bias_1")

def _debate_bias_2_viewpoint(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_bias_viewpoint(memory, action_functions,ThoughtState.BIAS_1_CROSS_EXAMINATION,"bias_2")

def _debate_bias_1_cross_examination(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_cross_examination(memory, action_functions,ThoughtState.BIAS_2_CROSS_EXAMINATION,"bias_1","bias_2")

def _debate_bias_2_cross_examination(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_cross_examination(memory, action_functions,ThoughtState.BIAS_1_CROSS_EXAMINATION_ANSWERS,"bias_2","bias_1")

def _debate_bias_1_cross_examination_answers(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_cross_examination_answers(memory, action_functions,ThoughtState.BIAS_2_CROSS_EXAMINATION_ANSWERS,"bias_1","bias_2")

def _debate_bias_2_cross_examination_answers(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_cross_examination_answers(memory, action_functions,ThoughtState.BIAS_1_DIFF_VIEWPOINT_SUMMARY,"bias_2","bias_1")

def _debate_bias_1_diff_viewpoint_summary(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_diff_viewpoint_summary(memory, action_functions,ThoughtState.BIAS_2_DIFF_VIEWPOINT_SUMMARY,"bias_1","bias_2")

def _debate_bias_2_diff_viewpoint_summary(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_diff_viewpoint_summary(memory, action_functions,ThoughtState.BIAS_1_CONCLUSION,"bias_2","bias_1")

def _debate_bias_1_conclusion(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_conclusion(memory, action_functions,ThoughtState.BIAS_2_CONCLUSION,"bias_1","bias_2")

def _debate_bias_2_conclusion(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return debate_bias_conclusion(memory, action_functions,ThoughtState.STORE_INFO,"bias_2","bias_1")

def _store_info(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return store_info(memory,action_functions,ThoughtState.END)


# The dictionary mapping ThoughtState enums to their corresponding functions
state_function_map: Dict[ThoughtState, Callable[[Dict, Callable[[str], str]], Tuple[ThoughtState, Dict]]] = {
    ThoughtState.SUMMARIZE_AS_BULLETS: _summarize_as_bullets,
    ThoughtState.FILTER_FOR_BIAS: _filter_for_bias,
    ThoughtState.REWRITE_FOR_BIAS: _rewrite_for_debate,
    ThoughtState.BIAS_1_VIEWPOINT: _debate_bias_1_viewpoint,
    ThoughtState.BIAS_2_VIEWPOINT: _debate_bias_2_viewpoint,
    ThoughtState.BIAS_1_CROSS_EXAMINATION: _debate_bias_1_cross_examination,
    ThoughtState.BIAS_2_CROSS_EXAMINATION: _debate_bias_2_cross_examination,
    ThoughtState.BIAS_1_CROSS_EXAMINATION_ANSWERS: _debate_bias_1_cross_examination_answers,
    ThoughtState.BIAS_2_CROSS_EXAMINATION_ANSWERS: _debate_bias_2_cross_examination_answers,
    ThoughtState.BIAS_1_DIFF_VIEWPOINT_SUMMARY: _debate_bias_1_diff_viewpoint_summary,
    ThoughtState.BIAS_2_DIFF_VIEWPOINT_SUMMARY: _debate_bias_2_diff_viewpoint_summary,
    ThoughtState.BIAS_1_CONCLUSION: _debate_bias_1_conclusion,
    ThoughtState.BIAS_2_CONCLUSION: _debate_bias_2_conclusion,
    ThoughtState.STORE_INFO: _store_info,
}


def get_init_debator_state(text_chunks:List[str],bias_tup_list:List[Tuple[str,str]])->List[Tuple[ThoughtState, Dict]]:
    """Initializes state/memory dictionaries for all bias values for all text chunks"""
    return [
        (ThoughtState.SUMMARIZE_AS_BULLETS, get_debator_biases_init_memory(text_chunk, bias_tup))
        for text_chunk, bias_tup in product(text_chunks, bias_tup_list)
    ]

def process_debator_state(initial_state: ThoughtState, 
                  memory: Dict, 
                  action_functions: ActionFunctions)->List[Tuple[ThoughtState, Dict]]:
    """ 
    Returns end state when complete - 
    if end state is given an error is thrown 
    """
    if initial_state == ThoughtState.END:
        return []
    try:
        # Retrieve the function for the current state
        state_function = state_function_map[initial_state]
        # Call the function with memory and the LLM function
        return state_function(memory, action_functions)
    
    except KeyError:
        raise KeyError(f"Unhandled state: {initial_state}. No function mapped for this ThoughtState.")
        return []