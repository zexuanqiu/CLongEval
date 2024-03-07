import os 
import argparse 
import jsonlines 
import sys 
import sys
sys.path.append('../')
from typing import List, Dict, Tuple
sys.setrecursionlimit(10000)
from post_processing_for_stacked_tasks import news_labeling_pattern_match
import re
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../model_checkpoints/internlm2-7b", trust_remote_code=True)


def news_labeling_scores_in_context(prediction, ground_truth) -> List[Tuple]:
    prediction_list = news_labeling_pattern_match(prediction.strip())
    ground_truth_list = []
    pattern = r"新闻\s(\d+)[：](\w+)"
    for cur_news in ground_truth.split("\n"):
        match = re.search(pattern, cur_news)
        ground_truth_list.append("，".join([match.group(1), match.group(2)]))
    
    score_1_set = set(prediction_list) & set(ground_truth_list)
    score_0_set = set(ground_truth_list) - score_1_set

    score_1_list = list(score_1_set)
    score_1_list = [(int(ele.split("，")[0]), 1.0) for ele in score_1_list]

    score_0_list = list(score_0_set)
    score_0_list = [(int(ele.split("，")[0]), 0.0) for ele in score_0_list]

    all_score_list = score_1_list + score_0_list #根据每个ele的大小排序
    sorted_score_list = sorted(all_score_list, key=lambda x: x[0])
    return sorted_score_list
    

dataset2metric = {
    "news_labeling": news_labeling_scores_in_context
}



def get_news_position(context:str, internlm2_length):
    news_list = context.split("\n\n")
    news_position_list = [] 
    for cur_news in news_list:
        start_idx = context.find(cur_news)
        start_token_length = len(tokenizer.encode(context[:start_idx], add_special_tokens=False))
        end_token_length = start_token_length + len(tokenizer.encode(cur_news, add_special_tokens=False))
        mean_pos_length = (start_token_length + end_token_length) / 2.
        news_position_list.append(mean_pos_length / float(internlm2_length))
    return news_position_list
        

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="kimichat", choices=["gpt4-turbo-128k", "moonshot-v1"])
    parser.add_argument("--dataset", type=str, default="news_labeling", choices=['news_labeling'])
    return parser.parse_args(args)

def scorer(dataset_name, predictions, answers) -> List[List[Tuple]]:
    all_scores = []
    assert len(predictions) == len(answers)
    for prediction, ground_truth in zip(predictions, answers):
        score = dataset2metric[dataset_name](prediction=prediction, ground_truth=ground_truth)
        all_scores.append(score)
    return all_scores

def get_id_len_prediction_answer(jsonl_file, id_length_dict, model_name):
    if not os.path.exists(jsonl_file):
        return [], [], []
    ids = []
    len_list = []
    predictions = [] 
    ground_truths = [] 
    with jsonlines.open(jsonl_file, 'r') as f_reader:
        for cur_json_obj in f_reader:
            cur_pred = cur_json_obj['response_{}'.format(model_name)]
            if "HTTP_ERROR" in cur_pred or "UNKNOW_ERROR" in cur_pred:  # for gpt4-turbo
                continue
            cur_ans = cur_json_obj['answer']
            ids.append(cur_json_obj['id'])
            len_list.append(id_length_dict[cur_json_obj['id']])
            predictions.append(cur_pred)
            ground_truths.append(cur_ans)
    return ids, len_list, predictions, ground_truths

def translate_pos_ratio(pos_ratio, interval_number):
    interval_width = 1. / interval_number
    position = int(pos_ratio / interval_width) + 1
    return "pos_{}".format(str(position))

def get_length_and_context(jsonl_file):
    id_length_dict = {}
    id_context_dict = {}
    with jsonlines.open(jsonl_file, 'r') as f_reader:
        for cur_json_obj in f_reader:
            id_length_dict[cur_json_obj['id']] = cur_json_obj['internlm2_length']
            id_context_dict[cur_json_obj['id']] = cur_json_obj['context']
    return id_length_dict, id_context_dict
    
    

if __name__ == "__main__":
    args = parse_args() 
    dataset2path = {"news_labeling": "../data/3-1_stacked_news_labeling/"}
    cur_dataset_dir = args.dataset
    infer_model_dir = os.path.join('../inference_results', args.model_name)
    
    print(cur_dataset_dir)
    small_file  = os.path.join(infer_model_dir, cur_dataset_dir, "small.jsonl")
    medium_file = os.path.join(infer_model_dir, cur_dataset_dir, "medium.jsonl")

    small_data_file = os.path.join(dataset2path[args.dataset], "small.jsonl")
    medium_data_file = os.path.join(dataset2path[args.dataset], "medium.jsonl")
    small_id_length_dict, small_id_context_dict = get_length_and_context(small_data_file)
    medium_id_length_dict, medium_id_context_dict = get_length_and_context(medium_data_file)
    id_length_dict = {**small_id_length_dict, **medium_id_length_dict}
    id_context_dict = {**small_id_context_dict, **medium_id_context_dict}

    print(len(small_id_length_dict), len(medium_id_length_dict), len(id_length_dict))

    small_ids, small_lengths, small_predicitons, small_answers = get_id_len_prediction_answer(small_file, id_length_dict, args.model_name)
    medium_ids, medium_lengths, medium_predicitons, medium_answers = get_id_len_prediction_answer(medium_file, id_length_dict, args.model_name)
    print(len(small_ids), len(medium_ids), len(small_ids+medium_ids))

    ids = small_ids + medium_ids 
    lengths = small_lengths + medium_lengths
    predictions = small_predicitons + medium_predicitons
    answers = small_answers + medium_answers

    all_scores = scorer(cur_dataset_dir, predictions, answers) # List[List[Tuple]]

    with jsonlines.open("{}_{}.jsonl".format(args.model_name, args.dataset), "w") as writer:
        for cur_id, length, sorted_scores_list, cur_prediction in zip(ids, lengths, all_scores, predictions):
            means_pos_ratio_list = get_news_position(id_context_dict[cur_id], id_length_dict[cur_id])
            if len(means_pos_ratio_list) != len(sorted_scores_list):
                print(means_pos_ratio_list)
                print(sorted_scores_list)
                exit()
            for cur_news_score_tuple, means_pos_ratio in zip(sorted_scores_list, means_pos_ratio_list):
                writer.write({"id": cur_id, "internlm2_length": length, "news_idx": cur_news_score_tuple[0], "pos_ratio": means_pos_ratio, "score": cur_news_score_tuple[1]})

