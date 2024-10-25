from simple_agent.prompt_maker import make_list_example, make_dict_example, make_prompt_template, prompt_from_template
from typing import Dict,Tuple, List

SUMMARIZE_AS_BULLETS_DESCRIPTION="""
Summarize the provided input as concise bullet points. Highlight key facts, main arguments, or essential details.
"""

SUMMARIZE_AS_BULLETS_LEADING_TOKEN="summary_bullets:"

SUMMARIZE_AS_BULLETS_EXAMPLE_1_INPUT="""
Cloud computing has revolutionized the IT industry. It allows businesses to scale their operations, 
improve data accessibility, and reduce costs. Key players include Amazon Web Services (AWS), Microsoft Azure, 
and Google Cloud Platform. The flexibility of pay-as-you-go models makes it appealing for startups and enterprises alike.
"""

SUMMARIZE_AS_BULLETS_EXAMPLE_1_OUTPUT=[
"Cloud computing transforming the IT industry.",
"Benefits: Scalability, improved data accessibility, cost reduction.",
"Major providers: AWS, Microsoft Azure, Google Cloud Platform.",
"Popular due to flexible pay-as-you-go models, appealing to both startups and enterprises."]

SUMMARIZE_AS_BULLETS_RESPONSE_TEMPLATE_ITEMS=[
"<key facts, main arguments, or essential detail>",
"<key facts, main arguments, or essential detail>",
"..."
]


RATE_KNOWLEDGE_BY_BIAS_PT="""
Rate the following fact, argument or detail in terms of importance or relevance to a |bias|. 
This fact, argument or detail is rated as not_relevant or relevant. Explain any reasoning for this
rating. 

Use the following format:

how_important_is_the_information:
  relevance_rating: <not_relevant or relevant>
  reasoning: <explaination of why the rating>

|summary_bullet|

how_important_is_the_information:
"""

REWRITE_KNOWLEDGE_BY_BIAS_PT="""
Rewrite the following fact, argument or detail to ephasize the perspective of a |bias|. 
Explain any reasoning for this rewrite. 

Use the following format:

perspective_rewrite:
  statement_rewrite: <rewrite of statement>
  reasoning: <explaination of why rewrite>

|summary_bullet|

perspective_rewrite:
"""

def get_summarize_bias_init_memory(orignal_text:str,bias:str)->Dict:
    return {'text_chunk':orignal_text,'bias':bias}

def get_summarize_as_bullets_prompt_and_labels(memory: Dict)->Tuple[str,list[str]]:
    input_keys=['text_chunk']
    example_1=make_list_example(SUMMARIZE_AS_BULLETS_EXAMPLE_1_INPUT,
                                SUMMARIZE_AS_BULLETS_LEADING_TOKEN,
                                SUMMARIZE_AS_BULLETS_EXAMPLE_1_OUTPUT)
    resp_template=make_list_example("",
                                    SUMMARIZE_AS_BULLETS_LEADING_TOKEN,
                                    SUMMARIZE_AS_BULLETS_RESPONSE_TEMPLATE_ITEMS)
    #resp_template:str,input_keys:List[str],leading_token:str
    prompt_template=make_prompt_template(SUMMARIZE_AS_BULLETS_DESCRIPTION,
                                         [example_1],
                                         resp_template,
                                         input_keys,
                                         SUMMARIZE_AS_BULLETS_LEADING_TOKEN)
    return prompt_from_template(prompt_template,memory),[SUMMARIZE_AS_BULLETS_LEADING_TOKEN.replace(":","")]

def get_rate_knowledge_by_bias_prompt_and_labels(memory: Dict)->Tuple[str,list[str]]:
    input_keys=['summary_bullet']
    pro_temp=RATE_KNOWLEDGE_BY_BIAS_PT.replace("|bias|",memory.get('bias',"||"))
    return prompt_from_template(pro_temp,memory),["relevance_rating"]

def get_rewrite_knowledge_by_bias_prompt_and_labels(memory: Dict)->Tuple[str,list[str]]:
    input_keys=['summary_bullet']
    pro_temp=REWRITE_KNOWLEDGE_BY_BIAS_PT.replace("|bias|",memory.get('bias',"||"))
    return prompt_from_template(pro_temp,memory) ,["statement_rewrite"]