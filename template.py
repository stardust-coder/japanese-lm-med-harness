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

#MedPaLM style
medpalm_four_choice_cot = Template('''### Instruction:
The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a single option from the four options as the final answer.
### Input:
{{instruction}}
{{input}}
### Response:
''')

medpalm_four_choice_cot_ja = Template('''### 指示：
以下は医学知識に関する多肢選択問題です。利用可能な情報を要約してから、段階的に解決してください。最終的な答えとして4つの選択肢のうちの1つを出力してください。
### 入力：
{{instruction}}
{{input}}
### 応答：
''')


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

##1shot
# medpalm_five_choice_cot = Template('''### Instruction:
# The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a single option from the five options as the final answer.
# ### Input:
# Which of the following is not a mandatory explanation to be provided to participants in human genome/gene analysis research?
# The purpose of the research, The freedom to consent, Methods for anonymity, Disadvantages of participation, Assurance of research results sharing
# ### Response:
# Assurance of research results sharing
# ### Input:
# {{instruction}}
# {{input}}
# ### Response:
# ''')

# medpalm_five_choice_cot_ja = Template('''### 指示：
# 以下は医学知識に関する多肢選択問題です。利用可能な情報を要約してから、段階的に解決してください。最終的な答えとして5つの選択肢のうちの1つを出力してください。
# ### 入力：
# ヒトゲノム・遺伝子解析研究の被験者に対する説明で必須でないのはどれか。
# 研究の目的, 同意の自由, 匿名化の方法, 参加による不利益, 研究成果還元の保証
# ### 応答：
# 研究成果還元の保証
# ### 入力：
# {{instruction}}
# {{input}}
# ### 応答：
# # ''')


##3shot
# medpalm_five_choice_cot = Template('''### Instruction:
# The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a single option from the five options as the final answer.
# ### Input:
# Which of the following is not a mandatory explanation to be provided to participants in human genome/gene analysis research?
# The purpose of the research, The freedom to consent, Methods for anonymity, Disadvantages of participation, Assurance of research results sharing
# ### Response:
# Assurance of research results sharing
# ### Input:
# A 57-year-old man lost consciousness and collapsed while working to remove sludge from a manhole at a sewage treatment plant. A colleague who entered to assist also suddenly lost consciousness and collapsed. Which of the following is the most likely cause? Select two.
# Oxygen deficiency, Hydrogen sulfide poisoning, Carbon monoxide poisoning, Carbon dioxide poisoning, Nitrogen dioxide poisoning
# ### Response:
# Oxygen deficiency, Hydrogen sulfide poisoning
# ### Input:
# A 28-year-old woman at 30 weeks of gestation has a fundal height of 22 cm and almost no amniotic fluid is detected on abdominal ultrasound examination. What is the most likely condition in the fetus?
# Esophageal atresia, Ventricular septal defect, Renal hypoplasia, Anorectal malformation, Fetal hydrops
# ### Response:
# Renal hypoplasia
# ### Input:
# {{instruction}}
# {{input}}
# ### Response:
# ''')

# medpalm_five_choice_cot_ja = Template('''### 指示：
# 以下は医学知識に関する多肢選択問題です。利用可能な情報を要約してから、段階的に解決してください。最終的な答えとして5つの選択肢のうちの1つを出力してください。
# ### 入力：
# ヒトゲノム・遺伝子解析研究の被験者に対する説明で必須でないのはどれか。
# 研究の目的, 同意の自由, 匿名化の方法, 参加による不利益, 研究成果還元の保証
# ### 応答：
# 研究成果還元の保証
# ### 入力：
# 57歳の男性。下水処理場のマンホール内で汚泥を外に搬出する作業を行っていたが、突然意識を失って倒れた。さらに救助しようとして中に入った同僚も急激に意識を失って倒れた。可能性が高いのはどれか。2つ選べ。
# 酸素欠乏症, 硫化水素中毒, 一酸化炭素中毒, 二酸化炭素中毒, 二酸化窒素中毒
# ### 応答：
# 酸素欠乏症, 硫化水素中毒
# ### 入力：
# 28歳の女性。妊娠30週。子宮底長は22cmで、腹部超音波検査で羊水はほとんど認めない。胎児で最も考えられるのはどれか。
# 食道閉鎖, 心室中隔欠損, 腎低形成, 鎖肛, 胎児水腫
# ### 応答：
# 腎低形成
# ### 入力：
# {{instruction}}
# {{input}}
# ### 応答：
# ''')


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
        if "ja" in name:
            return alpaca_ja
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
