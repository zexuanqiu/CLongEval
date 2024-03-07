import os 
import torch 
# from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm 
import numpy as np 
import random 
import argparse
import json
import jsonlines
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
from openai import OpenAI
import time 

def load_dataset(dataset_path):
    data_all = [] 
    with jsonlines.open(dataset_path, "r") as f_reader:
        for json_obj in f_reader:
            data_all.append(json_obj)
    return data_all


def get_pred_vllm(data, task_prompt, model_name, model_path, tp_size, gpu_memory_utilization, max_tokens, save_res_path):
    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(temperature=0.01,max_tokens=max_tokens, stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>, </s>"])

    model = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization,tensor_parallel_size=tp_size)
    existing_id_set = set() 
    print(save_res_path)
    if os.path.exists(save_res_path):
        with jsonlines.open(save_res_path, "r") as reader:
            for json_obj in reader:
                existing_id_set.add(json_obj['id'])

    # only eneumerate those sample that haven't been inferenecd 
    for idx, cur_sample in tqdm(enumerate(data)):
        if cur_sample['id'] not in existing_id_set:
            if 'query' not in cur_sample or len(cur_sample['query']) == 0: # 2-1, 3-1, 3-2
                input_to_model = task_prompt.format(cur_sample['context'])
            else: # 1-1, 1-2, 1-3, 1-4
                input_to_model = task_prompt.format(cur_sample['context'], cur_sample['query'])
                
            output = model.generate(input_to_model, sampling_params)[0]
            context_len = len(output.prompt_token_ids)
            cur_response = output.outputs[0].text

            with open(save_res_path, "a", encoding="utf-8") as f:
                json.dump({'id': cur_sample['id'], 'query': cur_sample['query'], 'answer': cur_sample['answer'], 'response_{}'.format(model_name): cur_response}, f, ensure_ascii=False)
                f.write("\n")
            print(idx, cur_response)
            

def get_pred_lmdeploy(data, task_prompt, model_name, model_path, tp_size, gpu_memory_utilization, max_tokens, save_res_path):
    from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
    if "20b" in model_name:
        print("Using InternLM2-20B.")
        if "medium" in save_res_path:
            backend_config = TurbomindEngineConfig(tp=tp_size, cache_max_entry_count=0.3,max_batch_size=2,
                            session_len=80000, rope_scaling_factor=3.0)
        elif "large" in save_res_path:
            backend_config = TurbomindEngineConfig(tp=tp_size, cache_max_entry_count=0.3,max_batch_size=2,
                            session_len=160000, rope_scaling_factor=3.0)
        else:
            backend_config = TurbomindEngineConfig(tp=tp_size, cache_max_entry_count=0.3,max_batch_size=2)
        gen_config= GenerationConfig(top_p=1.0, top_k=1, temperature=0.01, max_new_tokens=max_tokens)

    else:
        print("Using InternLM2-7B.")
        if "medium" in save_res_path:
            backend_config = TurbomindEngineConfig(tp=tp_size, cache_max_entry_count=0.3,max_batch_size=2,
                            session_len=80000, rope_scaling_factor=5.0)
        elif "large" in save_res_path:
            backend_config = TurbomindEngineConfig(tp=tp_size, cache_max_entry_count=0.3,max_batch_size=2,
                            session_len=160000, rope_scaling_factor=5.0)
        else:
            backend_config = TurbomindEngineConfig(tp=tp_size, cache_max_entry_count=0.3,max_batch_size=2)
        gen_config= GenerationConfig(top_p=1.0, top_k=1, temperature=0.01, max_new_tokens=max_tokens) 

    pipe = pipeline(model_path,
                    backend_config=backend_config)
    
    existing_id_set = set()
    if os.path.exists(save_res_path):
        with jsonlines.open(save_res_path, "r") as reader:
            for json_obj in reader:
               existing_id_set.add(json_obj['id'])

    # only eneumerate those sample that haven't been inferenecd 
    for idx, cur_sample in tqdm(enumerate(data)):
        if cur_sample['id'] not in existing_id_set:
            if 'query' not in cur_sample or len(cur_sample['query']) == 0: # 2-1, 3-1, 3-2
                input_to_model = task_prompt.format(cur_sample['context'])
            else: # 1-1, 1-2, 1-3, 1-4
                input_to_model = task_prompt.format(cur_sample['context'], cur_sample['query'])

            output = pipe(input_to_model, gen_config=gen_config)
            cur_response = output.text
            # print("Finished Reason: {}".format(output.finish_reason))
            with open(save_res_path, "a", encoding="utf-8") as f:
                json.dump({'id': cur_sample['id'], 'query': cur_sample['query'], 'answer': cur_sample['answer'], 'response_{}'.format(model_name): cur_response}, f, ensure_ascii=False)
                f.write("\n")
            print(idx, cur_response)


def get_pred_glm4(data, task_prompt, model_name, max_tokens, save_res_path):
    from zhipuai import ZhipuAI
    existing_id_set = set()
    if os.path.exists(save_res_path):
        with jsonlines.open(save_res_path, "r") as reader:
            for json_obj in reader:
               existing_id_set.add(json_obj['id'])
    api_key = "Your_Key"
    client = ZhipuAI(api_key=api_key)

    def try_glm4(client, max_tokens, input_to_model, count=0):
            cur_count = count
            try: 
                completion = client.chat.completions.create(
                model="glm-4",
                messages=[ 
                    {"role": "system", "content": "你是一个人工智能助手。你会为用户提供有帮助，准确的回答。"},
                    {"role": "user", "content": input_to_model}
                ],
                top_p=0.99,
                temperature=0.01, # same as kimichat, gpt4-turbo
                max_tokens=max_tokens)
                message = completion.choices[0].message
                cur_response = message.content
                return cur_response
            except Exception as e:
                print("Current attempt is {}, Error Message：{}".format(cur_count, e.__str__()))
                if count == 0:
                    return None
                else:
                    time.sleep(2)
                    return try_glm4(client, max_tokens, input_to_model, count=cur_count+1)                

    for idx, cur_sample in tqdm(enumerate(data)):
        if cur_sample['id'] not in existing_id_set:
            if 'query' not in cur_sample or len(cur_sample['query']) == 0: # 2-1, 3-1, 3-2
                input_to_model = task_prompt.format(cur_sample['context'])
            else: # 1-1, 1-2, 1-3, 1-4
                input_to_model = task_prompt.format(cur_sample['context'], cur_sample['query'])
            
            if idx % 10 == 0:
                time.sleep(3)
                print("{}, sleep 3 seconds.".format(idx))

            cur_response = try_glm4(client, max_tokens, input_to_model)
            if cur_response is None:
                print("The {}th sample request is incorrect. The sample ID is {}.".format(idx, cur_sample['id']))
                continue
            print(idx, cur_response)
            
            with open(save_res_path, "a", encoding="utf-8") as f:
                json.dump({'id': cur_sample['id'], 'query': cur_sample['query'], 'answer': cur_sample['answer'], 'response_{}'.format(model_name): cur_response}, f, ensure_ascii=False)
                f.write("\n")


def get_pred_moonshot(data, task_prompt, model_name, max_tokens, save_res_path):
    existing_id_set = set()
    if os.path.exists(save_res_path):
        with jsonlines.open(save_res_path, "r") as reader:
            for json_obj in reader:
               existing_id_set.add(json_obj['id'])

    client = OpenAI(
        api_key = "Your Key",
        base_url="https://api.moonshot.cn/v1",
    )

    def try_moonshot(client, max_tokens, input_to_model, count=0):
            cur_count = count
            try: 
                completion = client.chat.completions.create(
                model="moonshot-v1-128k",
                messages=[ 
                    {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你会为用户提供有帮助，准确的回答。"},
                    {"role": "user", "content": input_to_model}
                ],
                temperature=0.01,
                top_p=1.,
                max_tokens=max_tokens)
                message = completion.choices[0].message
                cur_response = message.content
                return cur_response
            except Exception as e:
                print("Current attempt is {}, Error Message：{}".format(cur_count, e.__str__()))
                if count == 0: # 0 means no more attempts will be made for this sample, just skip and return None
                    return None
                else:
                    time.sleep(2)
                    return try_moonshot(client, max_tokens, input_to_model, count=cur_count+1)                

    for idx, cur_sample in tqdm(enumerate(data)):
        if cur_sample['id'] not in existing_id_set:
            if 'query' not in cur_sample or len(cur_sample['query']) == 0: # 2-1, 3-1, 3-2
                input_to_model = task_prompt.format(cur_sample['context'])
            else: # 1-1, 1-2, 1-3, 1-4
                input_to_model = task_prompt.format(cur_sample['context'], cur_sample['query'])
            
            if idx % 10 == 0:
                time.sleep(3)
                print("{}, sleep 3 seconds".format(idx))

            cur_response = try_moonshot(client, max_tokens, input_to_model)
            if cur_response is None:
                print("The {}th sample request is incorrect. The sample ID is {}.".format(idx, cur_sample['id']))
                continue
            print(idx, cur_response)
            
            with open(save_res_path, "a", encoding="utf-8") as f:
                json.dump({'id': cur_sample['id'], 'query': cur_sample['query'], 'answer': cur_sample['answer'], 'response_{}'.format(model_name): cur_response}, f, ensure_ascii=False)
                f.write("\n")


def get_pred_gpt4(data, task_prompt, model_name, max_tokens, save_res_path):
    existing_id_set = set()
    if os.path.exists(save_res_path):
        with jsonlines.open(save_res_path, "r") as reader:
            for json_obj in reader:
               existing_id_set.add(json_obj['id'])
 
    client = OpenAI(
        api_key = "Your Key",
    )

    def try_gpt4(client, max_tokens, input_to_model, count=0):
            cur_count = count
            try: 
                completion = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[ 
                    {"role": "system", "content": "你是一个人工智能助手。你会为用户提供有帮助，准确的回答。"},
                    {"role": "user", "content": input_to_model}
                ],
                temperature=0.01,
                top_p=1.,
                max_tokens=max_tokens)
                message = completion.choices[0].message
                cur_response = message.content
                return cur_response
            except Exception as e:
                print("Current attempt is {}, Error Message：{}".format(cur_count, e.__str__()))
                if count == 0: # 0 means no more attempts will be made for this sample, just skip and return None
                    return None
                else:
                    time.sleep(2)
                    return try_gpt4(client, max_tokens, input_to_model, count=cur_count+1)                

    for idx, cur_sample in tqdm(enumerate(data)):
        if cur_sample['id'] not in existing_id_set:
            if 'query' not in cur_sample or len(cur_sample['query']) == 0: # 2-1, 3-1, 3-2
                input_to_model = task_prompt.format(cur_sample['context'])
            else: # 1-1, 1-2, 1-3, 1-4
                input_to_model = task_prompt.format(cur_sample['context'], cur_sample['query'])
            
            if idx % 10 == 0:
                time.sleep(3)
                print("{}, sleep 3 seconds".format(idx))

            cur_response = try_gpt4(client, max_tokens, input_to_model)
            if cur_response is None:
                print("The {}th sample request is incorrect. The sample ID is {}.".format(idx, cur_sample['id']))
                continue
            print(idx, cur_response)
            
            with open(save_res_path, "a", encoding="utf-8") as f:
                json.dump({'id': cur_sample['id'], 'query': cur_sample['query'], 'answer': cur_sample['answer'], 'response_{}'.format(model_name): cur_response}, f, ensure_ascii=False)
                f.write("\n")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, choices=['chatglm3-6b-32k', 'qwen-7b-32k', 'chinese-llama2-7b-64k', 'chinese-alpaca2-7b-64k', 'internlm2-7b-200k', 'internlm2-20b-200k', "moonshot-v1", "glm4-128k", "gpt4-turbo-128k"])
    parser.add_argument("--size", type=str, default=None, choices=['small.jsonl', "medium.jsonl", "large.jsonl"])
    parser.add_argument("--dataset_name", type=str, default=None, choices=['long_story_qa', 'long_conversation_memory','long_story_summarization', 'stacked_news_labeling', 'stacked_typo_detection', 'key_passage_retrieval', 'table_querying'])
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--gpus", type=str)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    tp_size = args.tensor_parallel_size 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_memory_utilization = args.gpu_memory_utilization
    seed_everything(42)

    dataset2path = json.load(open("config/dataset2path.json", "r", encoding="utf-8"))
    dataset_dir = dataset2path[args.dataset_name]
    dataset_path = os.path.join(dataset_dir, args.size)
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r", encoding="utf-8"))
    task_prompt = dataset2prompt[args.dataset_name]
    model2path = json.load(open("config/model2path.json", "r"))
    if "moonshot" not in args.model_name and "glm4" not in args.model_name and "gpt4" not in args.model_name:
        model_path = model2path[args.model_name]
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r", encoding="utf-8"))
    max_tokens = int(dataset2maxlen[args.dataset_name])
    

    if not os.path.exists("inference_results"):
        os.makedirs("inference_results")
    model_res_path = "inference_results/{}".format(args.model_name)
    save_res_dir = os.path.join(model_res_path, args.dataset_name)
    save_res_path = os.path.join(save_res_dir, args.size)
    if not os.path.exists(model_res_path):
        os.makedirs(model_res_path)
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    
    data_all = load_dataset(dataset_path)
    if "moonshot" in args.model_name:
        data_all_with_response = get_pred_moonshot(data=data_all, task_prompt=task_prompt, model_name=args.model_name, max_tokens=max_tokens, save_res_path=save_res_path)
        exit()
    if "gpt4" in args.model_name:
        data_all_with_response = get_pred_gpt4(data=data_all, task_prompt=task_prompt, model_name=args.model_name, max_tokens=max_tokens, save_res_path=save_res_path)
        exit()
    if "glm4" in args.model_name:
        data_all_with_response = get_pred_glm4(data=data_all, task_prompt=task_prompt, model_name=args.model_name, max_tokens=max_tokens, save_res_path=save_res_path)
        exit()

    if "internlm" in args.model_name:
        get_pred_func = get_pred_lmdeploy
    else:
        get_pred_func = get_pred_vllm
    data_all_with_response = get_pred_func(data=data_all, task_prompt=task_prompt,\
                                       model_name=args.model_name, model_path=model_path, tp_size=tp_size,
                                       gpu_memory_utilization=gpu_memory_utilization,
                                       max_tokens=max_tokens,save_res_path=save_res_path)



