import pandas as pd
import json
import copy
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from tqdm import tqdm
import pdb
from template import *
from vllm import LLM, SamplingParams
import difflib
import Levenshtein
import glob
import os
import random

# モデルのキャッシュパスの変更
cache_dir = "~/.cache" #Change if you want to store cache in different directory
os.environ['HF_HOME'] = cache_dir
os.makedirs("./result", exist_ok=True)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def parse():
    parser = argparse.ArgumentParser() # パーサーを作成
    
    parser.add_argument("--model_path", type=str, required = True)# オプションを指定
    parser.add_argument("--mode", type=str, required = False, default = "llm")# オプションを指定
    parser.add_argument("--prompt", type=str, required = True)# オプションを指定
    parser.add_argument("--data", type=str, required = True)# オプションを指定
    parser.add_argument("--lang", type=str, required = True)# オプションを指定
    parser.add_argument("--peft", type=str, required = False)# オプションを指定
    parser.add_argument("--quantize", action='store_true')# オプションを指定
    parser.add_argument("--shuffle", action='store_true')# オプションを指定
    
    parser.add_argument("--use_vllm", action='store_true')# オプションを指定
    parser.add_argument("--num_gpu", type=int, required = False, default=1)# オプションを指定
    

    # 解析
    args = parser.parse_args()
    # 入力値を表示
    # print(args.model_path)
    return args

def parse_model_name(path):
    if "/" in path:
        return path.split("/")[-1], path
    else:
        return path, path

def inference(args):
    model_name, model_id = parse_model_name(args.model_path)
    is_shuffle = "shuffle" if args.shuffle==True else "noshuffle"

    if args.use_vllm:
        #use vllm
        max_model_len = 2048
        llm = LLM(model=model_id,dtype='float16', tensor_parallel_size=args.num_gpu, max_model_len=max_model_len, quantization="AWQ") if args.quantize else LLM(model=model_id,dtype='float16', tensor_parallel_size=args.num_gpu, max_model_len=max_model_len)
        sampling_params = SamplingParams(temperature=1e-2, top_p=1.0, max_tokens=256) #greedy sampling
    else:
        #use model
        if args.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir,trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, cache_dir=cache_dir,device_map="auto",trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir,trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",cache_dir=cache_dir,trust_remote_code=True)

        if args.peft:
            peft_name = args.peft
            model = PeftModel.from_pretrained(
                model, 
                peft_name, 
                device_map="auto"
            )
            print("Loaded PEFT:",peft_name)
        model.eval()

    def generate(text):
        if args.use_vllm:
            output = llm.generate([text], sampling_params=sampling_params)[0]
            response = output.outputs[0].text
        else:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs,
                            do_sample=False, #決定論的
                            max_new_tokens=256,
                            )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def prepare_data(data_name, ln):
        # assert data_name in ["IgakuQA","IgakuQA2018","MedMCQA","MedQA","CardioExam","ClinicalQA25","sample","MMLU","JMMLU"]
        if "IgakuQA" in data_name:
            flag_ = False if data_name=="IgakuQA-v" else True
            if "20" in data_name:
                if data_name=="IgakuQA":
                    en_files = glob.glob("dataset/IgakuQA/data/*/*_translate.jsonl")
                    jp_files = glob.glob("dataset/IgakuQA/data/*/*-[ABCDEF]_jp.jsonl")
                else:
                    year = data_name.replace("IgakuQA","")
                    en_files = glob.glob("dataset/IgakuQA/data/"+year+"/*_translate.jsonl")
                    jp_files = glob.glob("dataset/IgakuQA/data/"+year+"/*-[ABCDEF]_jp.jsonl")
            else:
                en_files = glob.glob("dataset/IgakuQA/data/20*/*_translate.jsonl")
                jp_files = glob.glob("dataset/IgakuQA/data/20*/*-[ABCDEF]_jp.jsonl")

            en_files.sort()
            jp_files.sort()
            jsonl_data = []
            files = en_files if ln=="en" else jp_files
            for file, jp_file in zip(files,jp_files):
                with open(file) as f, open(jp_file) as g:
                    tmp_ = f.readlines()
                    jp_tmp_ = g.readlines()
                    for l1,l2 in zip(tmp_,jp_tmp_):
                        j1 = json.loads(l1)
                        j2 = json.loads(l2)
                        
                        if j2["text_only"] == flag_ and len(j2["choices"]) == 5:
                            jsonl_data.append(j1)
            return jsonl_data
        
        elif data_name == "ClinicalQA25":
            jsonl_data = []
            with open(f"dataset/ClinicalQA25/clinicalqa25_{ln}.jsonl") as f:
                tmp_ = f.readlines()
                for l in tmp_:
                    j = json.loads(l)
                    j['problem_text'] = j['question']
                    j["choices"] = ["","","","",""]
                    del j['question']
                    jsonl_data.append(j)
            return jsonl_data

        else:
            if "JMMLU-" in data_name:
                topic = data_name.replace("JMMLU-","")
                if not os.path.exists(f"dataset/JMMLU/{topic}.jsonl"):#only first inference to generate jsonl:
                    df_ = pd.read_csv(f"dataset/JMMLU/{topic}.csv", header=None, names=['problem_text', 'choice1','choice2','choice3','choice4','correct_choice'])
                    choices = []
                    answers = []
                    for i,v in df_.iterrows():
                        choice = [v["choice1"],v["choice2"],v["choice3"],v["choice4"]]
                        letter_to_ind = {"A":0,"B":1,"C":2,"D":3}
                        answer = choice[letter_to_ind[v["correct_choice"]]]
                        choices.append(choice)
                        answers.append(answer) 
                    df_["choices"] = choices
                    df_["answer"] = answers
                    df_.to_json(f"dataset/JMMLU/{topic}.jsonl", orient='records', force_ascii=False, lines=True)
            
            elif "MMLU-" in data_name:
                topic = data_name.replace("MMLU-","")
                if not os.path.exists(f"dataset/MMLU/{topic}.jsonl"):#only first inference to generate jsonl:
                    df_ = pd.read_csv(f"dataset/MMLU/{topic}_test.csv", header=None, names=['problem_text', 'choice1','choice2','choice3','choice4','correct_choice'])
                    choices = []
                    answers = []
                    for i,v in df_.iterrows():
                        choice = [v["choice1"],v["choice2"],v["choice3"],v["choice4"]]
                        letter_to_ind = {"A":0,"B":1,"C":2,"D":3}
                        answer = choice[letter_to_ind[v["correct_choice"]]]
                        choices.append(choice)
                        answers.append(answer) 
                    df_["choices"] = choices
                    df_["answer"] = answers
                    df_.to_json(f"dataset/MMLU/{topic}.jsonl", orient='records', force_ascii=False, lines=True)
            else:
                pass

            if data_name == "sample":
                files = glob.glob("dataset/IgakuQA/data/2018/112-A.jsonl")
            elif data_name == "MedMCQA":
                files = glob.glob(f"dataset/MedMCQA/medmcqa_{ln}.jsonl")
            elif data_name == "MedQA":
                files = glob.glob(f"dataset/MedQA/usmleqa_{ln}.jsonl")
            elif data_name == "CardioExam":
                files = glob.glob(f"dataset/CardioExam/cardioexam_{ln}.jsonl")
            elif data_name == "JMMLU":
                assert ln == "ja"
                files = glob.glob(f"dataset/JMMLU/*.jsonl")
            elif "JMMLU" in data_name:
                assert ln == "ja"
                category = data_name.replace("JMMLU-","")
                files = glob.glob(f"dataset/JMMLU/{category}.jsonl")
            elif data_name == "MMLU":
                assert ln == "en"
                files = glob.glob(f"dataset/MMLU/*.jsonl")
            elif "MMLU" in data_name:
                assert ln == "en"
                category = data_name.replace("MMLU-","")
                files = glob.glob(f"dataset/MMLU/{category}.jsonl")

            files.sort()
            jsonl_data = []
           
            for file in files:
                with open(file) as f:
                    tmp_ = f.readlines()
                    for l in tmp_:
                        j = json.loads(l)
                        jsonl_data.append(j)
            return jsonl_data


    test_data = prepare_data(args.data, args.lang)
    print("len of data:", len(test_data))
    assert len(test_data) != 0

    result = []
    id_ = 0
    for item in tqdm(test_data):
        try:
            problem_id = item["problem_id"]
        except:
            id_ += 1
            problem_id = str(id_)
        print(problem_id)
        print(item)
        if "Igaku" in args.data:
            question = item["problem_text"] if args.lang == "ja" else item["problem_text_en"]
            choices = item["choices"] if args.lang == "ja" else item["choices_en"]
            
        else:
            question = item["problem_text"]
            choices = item["choices"]
        

        if (len(question) <= 1024 and args.lang=="ja") or (len(question) <= 2048 and args.lang=="en"): #depends on GPU capacity
            if "Igaku" in args.data or args.data=="sample": #IgakuQAはanswerが記号なので, 選択肢に修正.
                assert len(choices) == 5
                options = {"a":choices[0],"b":choices[1],"c":choices[2],"d":choices[3],"e":choices[4]} #dict
                answer = [options[ind] for ind in item["answer"]]
                answer = ",".join(answer)
            else:
                answer = item["answer"]

            if args.shuffle: #this should be latter then the process above
                random.shuffle(choices)

            prompt_template = prepare_prompt_template(args.prompt) #liquid.Template
            if "meditron_four_choice" in args.prompt:
                prompt = prompt_template.render(instruction=question,input=f"(A){choices[0]}\n(B){choices[1]}\n(C){choices[2]}\n(D){choices[3]}")
            if "meditron_five_choice" in args.prompt:
                prompt = prompt_template.render(instruction=question,input=f"(A){choices[0]}\n(B){choices[1]}\n(C){choices[2]}\n(D){choices[3]}\n(E){choices[4]}")    
            else:          
                prompt = prompt_template.render(instruction=question,input=",".join(choices))
            full_response = generate(prompt)

            # ###(old)Zero-shot inference            
            # if args.lang=="ja":
            #     response = full_response.split("### 応答：\n")[-1]
            # else:
            #     if "Answer" in full_response:
            #         response = full_response.split("### Answer")[-1]
            #         response = response.split("###")[0]
            #     elif "Response" in full_response:
            #         response = full_response.split("### Response")[-1]
            #         response = response.split("###")[0]
            #     else:
            #         response = full_response.split("### Response")[-1]
            
            ###k-shot inference
            k = 0
            if "### 応答：\n" in full_response:
                response = full_response.split("### 応答：\n")[1+k]
            elif "### Answer:" in full_response:
                response = full_response.split("### Answer:")[1+k]
            elif "### Response:" in full_response:
                response = full_response.split("### Response:")[1+k]
            else:
                response = full_response
            response = response.strip()
            response = response.split("\n")[0]
            response = response.strip()
            print(response)

            result.append([problem_id,question,choices,answer,full_response,response])
   
    result_df = pd.DataFrame(result, columns=["problem_id","question","choices","answer","full_response","response"])
    result_df.to_json(f"result/{model_name}-{args.data}-{args.prompt}-{is_shuffle}.jsonl", orient='records', force_ascii=False, lines=True)


def evaluate(args):
    #Gestalt score
    def gestalt_dist(word1: str, word2: str) -> float:
        return difflib.SequenceMatcher(None, word1, word2).ratio()
    
    def jaro_winkler_dist(word1: str, word2: str) -> float:
        return Levenshtein.jaro_winkler(word1, word2)

    model_name, model_id = parse_model_name(args.model_path)    
    is_shuffle = "shuffle" if args.shuffle==True else "noshuffle"
    with open(f"result/{model_name}-{args.data}-{args.prompt}-{is_shuffle}.jsonl",encoding="utf-8") as f:
        result = [json.loads(l) for l in f.readlines()]
    
    new_result = []
    
    exact_match_score = 0 #Exact Match
    gestalt_match_score = 0 #Gestalt Match
    for res in result:
        try:
            new_res = copy.deepcopy(res)
            question_id = res["problem_id"]
            correct_answer = res["answer"]
            correct_answer_list = correct_answer.split(",")
            response = res["response"]
            choices = res["choices"]
            num_choices = len(correct_answer_list)

            #Cleansing
            correct_answer = correct_answer.replace(".","").strip()
            response = response.replace(".","").strip()

            #Exact Match
            if correct_answer == response:
                new_res["exact_match"] = 1
                exact_match_score += 1
            else:
                new_res["exact_match"] = 0

            gestalt_scores = [(gestalt_dist(c,response)+gestalt_dist(response,c))/2 for c in choices]
            new_res["gestalt_score"] = gestalt_scores

            choice_selected = []
            choices_temp = copy.deepcopy(choices)
            gestalt_scores_temp = copy.deepcopy(gestalt_scores)
            
            for _ in range(num_choices):
                ind_selected = gestalt_scores_temp.index(max(gestalt_scores_temp))
                choice_selected_temp = choices_temp[ind_selected]
                choices_temp.remove(choice_selected_temp)
                gestalt_scores_temp.pop(ind_selected)

                choice_selected.append(choice_selected_temp)
            
            correct_answer_list.sort()
            choice_selected.sort()
            if correct_answer_list == choice_selected:
                new_res["gestalt_match"] = 1
                gestalt_match_score += 1
            else:
                new_res["gestalt_match"] = 0
            new_result.append(new_res)
        except:
            print("Error in ", question_id)
    
    print("Exact Match ①:", exact_match_score, "/", len(result), "=", exact_match_score/len(result))
    print("Gestalt Match:", gestalt_match_score, "/", len(result), "=", gestalt_match_score/len(result))

    with open(f"result/{model_name}-{args.data}-{args.prompt}-{is_shuffle}_eval.jsonl", 'w', newline='\n',encoding="utf-8") as g:
        for l in new_result:
            g.writelines(json.dumps(l, ensure_ascii=False))
            g.write("\n")

    
    # if lang == "jp":
    #     #,や，や.や．や。の統一
    #     result["choices"] = result["choices"].apply(lambda x: x.replace("，","**TOUTEN**").replace("、","**TOUTEN**").replace("．","**KUTEN**").replace(".","**KUTEN**").strip())
    #     result["answer"] = result["answer"].apply(lambda x: x.replace("，","**TOUTEN**").replace("、","**TOUTEN**").replace("．","**KUTEN**").replace(".","**KUTEN**").strip())
    #     result["response"] = result["response"].apply(lambda x: x.replace("，","**TOUTEN**").replace("、","**TOUTEN**").replace("．","**KUTEN**").replace(".","**KUTEN**").strip())
    
    # ###数字のカンマを置換
    # for num in [1,2,3,4,5,6,7,8,9]:
    #     result["choices"] = result["choices"].apply(lambda x: x.replace(f"{str(num)},000",f"{str(num)}000"))
    #     result["answer"] = result["answer"].apply(lambda x: x.replace(f"{str(num)},000",f"{str(num)}000"))
    #     result["response"] = result["response"].apply(lambda x: x.replace(f"{str(num)},000",f"{str(num)}000"))


if __name__=="__main__":
    args = parse()
    print("Start inference...")
    inference(args)
    print("Start evaluation...")
    evaluate(args)
