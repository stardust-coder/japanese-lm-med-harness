from liquid import Template


alpaca = Template('''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{{instruction}}
### Input:
{{input}}
### Response:
''')

alpaca_ja = Template('''以下は，タスクを説明する指示と，文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。
### 指示：
{{instruction}}
### 入力：
{{input}}
### 応答：
''')

#MedPaLM2 style
medpalm_five_choice_cot = Template('''### Instruction:
The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a single option from the five options as the final answer.
### Input:
{{instruction}}
{{input}}
### Response:
''')

medpalm_five_choice_cot_ja = Template('''### 指示：
以下は医学知識に関する多肢選択問題です。利用可能な情報を要約してから、段階的に解決してください。最終的な答えとして5つの選択肢のうちの1つを出力してください。
### 入力：
{{instruction}}
{{input}}
### 応答：
''')

#Alpaca style
alpaca_med_five_choice_cot = Template('''The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a single option from the five options as the final answer.
### Instruction:
{{instruction}}
### Input:
{{input}}
### Response:
''')

alpaca_med_five_choice_cot_ja = Template('''以下は医学知識に関する多肢選択問題です。利用可能な情報を要約してから、段階的に解決してください。最終的な答えとして5つの選択肢のうちの1つを出力してください。
### 指示：
{{instruction}}
### 入力：
{{input}}
### 応答：
''')

meditron_four_choice =  Template('''
<|im_start|> system
You are a medical doctor answering real-world medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiple-choice question. Select one correct answer from A to D. Base your answer on the current and standard practices referenced in medical guidelines.<|im_end|>
<|im_start|> question
Question: {{instruction}}
Options:
{{input}}
<|im_start|> answer

''')

meditron_five_choice =  Template('''
<|im_start|> system
You are a medical doctor answering real-world medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiple-choice question. Select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines.<|im_end|>
<|im_start|> question
Question: {{instruction}}
Options:
{{input}}<|im_end|>
<|im_start|> answer

''')




def prepare_prompt_template(name):
    if name == "alpaca":
        return alpaca
    elif name == "alpaca_ja":
        return alpaca_ja
    elif name == "medpalm_five_choice_cot":
        return medpalm_five_choice_cot
    elif name == "medpalm_five_choice_cot_ja":
        return medpalm_five_choice_cot_ja
    elif name == "alpaca_med_five_choice_cot":
        return alpaca_med_five_choice_cot
    elif name == "alpaca_med_five_choice_cot_ja":
        return alpaca_med_five_choice_cot_ja
    elif name == "meditron_four_choice":
        return meditron_four_choice
    elif name == "meditron_five_choice":
        return meditron_five_choice
    else:
        return alpaca
    






# general_cot_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

# general_cot = Template('''
# Here is the question:
# {{question}}

# Here are the potential choices:
# {{options}}

# Please think step-by-step and generate your output in json:
# ''')

# general_medrag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

# general_medrag = Template('''
# Here are the relevant documents:
# {{context}}

# Here is the question:
# {{question}}

# Here are the potential choices:
# {{options}}

# Please think step-by-step and generate your output in json:
# ''')

# meditron_cot = Template('''
# ### User:
# Here is the question:
# ...

# Here are the potential choices:
# A. ...
# B. ...
# C. ...
# D. ...
# X. ...

# Please think step-by-step and generate your output in json.

# ### Assistant:
# {"step_by_step_thinking": ..., "answer_choice": "X"}

# ### User:
# Here is the question:
# {{question}}

# Here are the potential choices:
# {{options}}

# Please think step-by-step and generate your output in json.

# ### Assistant:
# ''')

# meditron_medrag = Template('''
# Here are the relevant documents:
# {{context}}

# ### User:
# Here is the question:
# ...

# Here are the potential choices:
# A. ...
# B. ...
# C. ...
# D. ...
# X. ...

# Please think step-by-step and generate your output in json.

# ### Assistant:
# {"step_by_step_thinking": ..., "answer_choice": "X"}

# ### User:
# Here is the question:
# {{question}}

# Here are the potential choices:
# {{options}}

# Please think step-by-step and generate your output in json.

# ### Assistant:
# ''')
