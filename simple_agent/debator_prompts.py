from simple_agent.prompt_maker import make_list_example, make_dict_example, make_prompt_template, prompt_from_template
from typing import Dict,Tuple, List


DEBATE_BIAS_VIEWPOINT_PT="""
Provide a biased viewpoint for a |bias_x|.

Given this statement:
|statement_rewrite|

Use the following format for the response:

START:
  viewpoint_bias_x: <statement about key points and arguments>
END

When providing the viewpoint about the statement provide the key points and arguments.

Ensure that the key viewpoint_bias_x is used and the symbols <,> are not used.

START:
"""

def get_debate_bias_viewpoint_prompt_and_labels(memory: Dict,bias_key:str)->Tuple[str,list[str]]:
    """
    bias_key could be either bias_1 or 2
    """
    pro_temp=DEBATE_BIAS_VIEWPOINT_PT.replace('bias_x',bias_key)
    input_keys=['statement_rewrite']
    pro_temp=pro_temp.replace(f"|{bias_key}|",memory.get(bias_key,"||"))
    return prompt_from_template(pro_temp,memory),[f"viewpoint_{bias_key}"]

DEBATE_BIAS_CROSS_EXAMINATION_PT="""
Provide questions from one point of view regarding a different point of view. 

Given this statement:
|statement_rewrite|

From the perspective of a |bias_y|, what questions would you have for someone with the following viewpoint:
|viewpoint_bias_x|

Use the following format for the response:

START:
  cross_examination_questions_bias_y: <questions examining the viewpoint>
END

When providing questions ensure that the key cross_examination_questions_bias_y is used and the symbols <,> are not used.

START:
"""

def get_debate_bias_cross_examination_prompt_and_labels(memory: Dict,bias_key_1:str,bias_key_2:str)->Tuple[str,list[str]]:
    """
    bias_key could be either bias_1 or 2
    """
    pro_temp=DEBATE_BIAS_CROSS_EXAMINATION_PT.replace('bias_x',bias_key_1)
    pro_temp=pro_temp.replace('bias_y',bias_key_2)
    input_keys=["statement_rewrite",f'viewpoint_{bias_key_1}']
    pro_temp=pro_temp.replace(f"|{bias_key_2}|",memory.get(bias_key_2,"||"))
    return prompt_from_template(pro_temp,memory),[f"cross_examination_questions_{bias_key_2}"]


DEBATE_BIAS_CROSS_EXAMINATION_ANSWERS_PT="""
Provide answers from the perspective of someone with the viewpoint below

Given this statement:
|statement_rewrite|

The following viewpoint is held by a |bias_x|:
|viewpoint_bias_x|

The following questions came up:
|cross_examination_questions_bias_y|

From the perspective of a |bias_x| how would you answer these questions?

Use the following format for the response:

START:
  cross_examination_answers_bias_x: <answers to the questions>
END

Speak in the first person as though this is your perspective.
When providing your answer esure that the key cross_examination_answers_bias_x is used and the symbols <,> are not used.

START:
"""

def get_debate_bias_cross_examination_answers_prompt_and_labels(memory: Dict,bias_key_1:str,bias_key_2:str)->Tuple[str,list[str]]:
    """
    bias_key could be either bias_1 or 2
    """
    pro_temp=DEBATE_BIAS_CROSS_EXAMINATION_ANSWERS_PT.replace('bias_x',bias_key_1)
    pro_temp=pro_temp.replace('bias_y',bias_key_2)
    input_keys=['statement_rewrite',f'viewpoint_{bias_key_1}',f'cross_examination_questions_{bias_key_2}']
    pro_temp=pro_temp.replace(f"|{bias_key_1}|",memory.get(bias_key_1,"||"))
    return prompt_from_template(pro_temp,memory),[f"cross_examination_answers_{bias_key_1}"]

BIAS_DIFF_VIEWPOINT_SUMMARY_PT="""
Draft a summary of viewpoint that has been cross examined.

Given this statement:
|statement_rewrite|

A |bias_y| has this viewpoint:
|viewpoint_bias_y|

You asked the following questions as a |bias_x| regarding the different viewpoint:
|cross_examination_questions_bias_x|

The other person responded to those questions like this:
|cross_examination_answers_bias_y|

From the perspective of a |bias_x| how would you summarize this other viewpoint
Speak in the first person as though this is your perspective.

Use the following format for the response:

START:
  diff_viewpoint_summary_bias_y: <summary of other viewpoint after cross examination>
END

When providing your summary of the viewpoint after cross examination ensure that the key diff_viewpoint_summary_bias_y is used and the symbols <,> are not used.

START:
"""


def get_debate_bias_diff_viewpoint_summary_prompt_and_labels(memory: Dict,bias_key_1:str,bias_key_2:str)->Tuple[str,list[str]]:
    """
    bias_key could be either bias_1 or 2
    """
    pro_temp=BIAS_DIFF_VIEWPOINT_SUMMARY_PT.replace('bias_x',bias_key_1)
    pro_temp=pro_temp.replace('bias_y',bias_key_2)
    input_keys=['statement_rewrite',
                f'viewpoint_{bias_key_2}',
                f'cross_examination_questions_{bias_key_1}',
                f'cross_examination_answers_{bias_key_2}']
    pro_temp=pro_temp.replace(f"|{bias_key_1}|",memory.get(bias_key_1,"||")).replace(f"|{bias_key_2}|",memory.get(bias_key_2,"||"))
    return prompt_from_template(pro_temp,memory),[f"diff_viewpoint_summary_{bias_key_2}"]

DEBATE_BIAS_CONCLUSION_PT="""
Draft a conclusion with the viewpoint of a |bias_x|.

Given this statement:
|statement_rewrite|

You had the following viewpoint:
|viewpoint_bias_x|

While a |bias_y| has this differing viewpoint:
|diff_viewpoint_summary_bias_y|

From the perspective of a |bias_x| what conclusions would you make regarding the statement and the differing viewpoint.
Speak in the first person as though this is your perspective.

Use the following format for the response:

START:
  debate_conclusions_bias_x: <concluding statements regarding the debate>
END

When providing your conclusion ensure that the key debate_conclusions_bias_x is used and the symbols <,> are not used.

START:
"""

def get_debate_bias_conclusion_prompt_and_labels(memory: Dict,bias_key_1:str,bias_key_2:str)->Tuple[str,list[str]]:
    """
    bias_key could be either bias_1 or 2
    """
    pro_temp=DEBATE_BIAS_CONCLUSION_PT.replace('bias_x',bias_key_1)
    pro_temp=pro_temp.replace('bias_y',bias_key_2)
    input_keys=['statement_rewrite',f'viewpoint_{bias_key_1}',f"diff_viewpoint_summary_{bias_key_2}"]
    pro_temp=pro_temp.replace(f"|{bias_key_1}|",memory.get(bias_key_1,"||")).replace(f"|{bias_key_2}|",memory.get(bias_key_2,"||"))
    return prompt_from_template(pro_temp,memory),[f"debate_conclusions_{bias_key_1}"]

def get_debator_biases_init_memory(orignal_text:str,bias_tuple:(str,str))->Dict:
    bias_1=bias_tuple[0]
    bias_2=bias_tuple[1]
    debate_bias=f"a heated debate between a {bias_1} and a {bias_2}, focusing on two distinct sides"
    return {'text_chunk':orignal_text,'bias_1':bias_1,'bias_2':bias_2,'bias':debate_bias}



