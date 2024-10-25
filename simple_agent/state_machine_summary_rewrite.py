"""
Pure functions for transitioning state. 
All functions call pure function but can take impure functions as args.
"""

from typing import Callable, Dict, Tuple, List
from itertools import product
from action_functions import ActionFunctions
from thought_state_functions import ThoughtState, summarize_as_bullets, filter_for_bias, rewrite_for_bias, store_info
from simple_agent.summarize_prompts import get_summarize_bias_init_memory

# Function placeholders for each state
def _summarize_as_bullets(memory: Dict, action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return summarize_as_bullets(memory, action_functions, ThoughtState.FILTER_FOR_BIAS)


def _filter_for_bias(memory: Dict, action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return filter_for_bias(memory, action_functions, ThoughtState.REWRITE_FOR_BIAS)


def _rewrite_for_bias(memory: Dict, action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return rewrite_for_bias(memory, action_functions,ThoughtState.STORE_INFO)


def _store_info(memory: Dict,action_functions: ActionFunctions) -> List[Tuple[ThoughtState, Dict]]:
    return store_info(memory,action_functions,ThoughtState.END)


# The dictionary mapping ThoughtState enums to their corresponding functions
state_function_map: Dict[ThoughtState, Callable[[Dict, Callable[[str], str]], Tuple[ThoughtState, Dict]]] = {
    ThoughtState.SUMMARIZE_AS_BULLETS: _summarize_as_bullets,
    ThoughtState.FILTER_FOR_BIAS: _filter_for_bias,
    ThoughtState.REWRITE_FOR_BIAS: _rewrite_for_bias,
    ThoughtState.STORE_INFO: _store_info,
}

def get_init_summarizer_state(text_chunks:List[str],bias_list:List[str])->List[Tuple[ThoughtState, Dict]]:
    """Initializes state/memory dictionaries for all bias values for all text chunks"""
    return [
        (ThoughtState.SUMMARIZE_AS_BULLETS, get_summarize_bias_init_memory(text_chunk, bias))
        for text_chunk, bias in product(text_chunks, bias_list)
    ]

def process_summarizer_rewrite_state(initial_state: ThoughtState, 
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

