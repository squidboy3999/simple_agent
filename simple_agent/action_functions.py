"""
Action functions is a class that has variables that map to functions with side 
effects so these can be passed around without reimplimenting side effects in other 
locations in the program.
"""
from typing import Callable, List
class ActionFunctions:
    """
    Mostly for storing impure funtions
    """
    def __init__(self, 
                 call_llm: Callable[[str], str], 
                 store_info: Callable[[str], str],
                 logger: Callable[[str], str]):
        """
        Initializes the StateFunction class by setting the functions to instance variables.
        """
        self.call_llm = call_llm
        self.store_info = store_info
        self.logger=logger