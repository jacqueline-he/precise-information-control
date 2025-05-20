DRAFT_BIRTHPLACE_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a factual assistant. 
<|eot_id|><|start_header_id|>user<|end_header_id|>
Name some {occupation}s born in {loc}. 
Return your output in numbered list form (enumerated as '1. <entity 1>\n2. <entity 2>\n3...'), and with no other explanation or additional information."
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

VERIFY_BIRTHPLACE_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, factual, and cautious assistant.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Where was the {occupation} {entity} born? 
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

INSTR_BIRTHPLACE_PROMPT = """
Who are some {occupation}s born in {loc}? Please answer in full sentences in an informative way.
"""


DRAFT_QAMPARI_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a factual assistant. 
<|eot_id|><|start_header_id|>user<|end_header_id|>
Provide a list of accurate entities for the given question. Separate answers by commas. Return only entities, and make sure you don't include any other explanation or additional information.
Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

VERIFY_QAMPARI_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a careful and highly factual assistant. Your task is to critically evaluate the factual accuracy of an answer to a given question based on your knowledge of Wikipedia. Do not speculate. Only respond with "TRUE" if the answer is fully correct, or "FALSE" if any part of it is incorrect, misleading, or unverifiable.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Evaluate the following:

{question} One answer is {answer}. TRUE or FALSE? Respond ONLY with TRUE or FALSE.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
