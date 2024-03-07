import os 
import argparse 
import jsonlines 
import sys 
sys.path.append('../')
sys.setrecursionlimit(10000)

from metrics import (
    qa_f1_zh_score,
    rouge_zh_score,
    news_labeling_zh_score,
    typo_detection_zh_score,
    retrieval_zh_edit_score,
    table_querying_match_score,
)

dataset2metric = {
    "long_story_qa": qa_f1_zh_score,
    "long_conversation_memory": qa_f1_zh_score, 
    "long_story_summarization": rouge_zh_score,
    "stacked_news_labeling": news_labeling_zh_score,
    "stacked_typo_detection": typo_detection_zh_score,
    "key_passage_retrieval": retrieval_zh_edit_score,
    "table_querying": table_querying_match_score
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, choices=['chatglm3-6b-32k', 'qwen-7b-32k', 'chinese-llama2-7b-64k', 'chinese-alpaca2-7b-64k', 'internlm2-7b-200k', 'internlm2-20b-200k', "moonshot-v1", "glm4-128k", "gpt4-turbo-128k"])

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--interval_number", type=int, default=6)
    parser.add_argument("--min_len", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=60000)
    return parser.parse_args(args)

def position_scorer(dataset_name, predictions, answers):
    all_scores = []
    assert len(predictions) == len(answers)
    for prediction, ground_truth in zip(predictions, answers):
        score = dataset2metric[dataset_name](prediction=prediction, ground_truth=ground_truth)
        all_scores.append(score * float(100))
    return all_scores

def get_id_prediction_answer(jsonl_file, model_name):
    if not os.path.exists(jsonl_file):
        return [], [], []
    ids = []
    predictions = [] 
    ground_truths = [] 
    with jsonlines.open(jsonl_file, 'r') as f_reader:
        for cur_json_obj in f_reader:
            cur_pred = cur_json_obj['response_{}'.format(model_name)]
            if "HTTP_ERROR" in cur_pred or "UNKNOW_ERROR" in cur_pred:
                # Due to network errors or other issues, there were a few instances where results could not be generated using gpt4-turbo-128k even after calling api 5 times. We have ignored the scoring calculations for these samples.
                continue
            else:
                cur_ans = cur_json_obj['answer']
            ids.append(cur_json_obj['id'])
            predictions.append(cur_pred)
            ground_truths.append(cur_ans)
    return ids, predictions, ground_truths

def translate_pos_ratio(pos_ratio, interval_number:int):
    interval_width = 1. / interval_number
    position = int(pos_ratio / interval_width) + 1
    return "pos_{}".format(str(position))

if __name__ == "__main__":
    args = parse_args() 
    cur_dataset_dir = args.dataset
    min_len = args.min_len 
    max_len = args.max_len
    scores = []
    infer_model_dir = os.path.join('../inference_results', args.model_name)
    interval_number = args.interval_number

    id_score_dict = {}
    small_file  = os.path.join(infer_model_dir, cur_dataset_dir, "small.jsonl")
    medium_file = os.path.join(infer_model_dir, cur_dataset_dir, "medium.jsonl")
    large_file = os.path.join(infer_model_dir, cur_dataset_dir, "large.jsonl")

    small_ids, small_predicitons, small_answers = get_id_prediction_answer(small_file, args.model_name)
    medium_ids, medium_predicitons, medium_answers = get_id_prediction_answer(medium_file, args.model_name)
    large_ids, large_predicitons, large_answers = get_id_prediction_answer(large_file, args.model_name)

    ids = small_ids + medium_ids + large_ids
    predictions = small_predicitons + medium_predicitons + large_predicitons
    answers = small_answers + medium_answers + large_answers 

    scores = position_scorer(cur_dataset_dir, predictions, answers)
    for cur_id, cur_score in zip(ids, scores):
        id_score_dict[cur_id] = cur_score

    positon_scores = {}
    with jsonlines.open("./data_position_info/{}_with_pos.jsonl".format(cur_dataset_dir), 'r') as f_reader:
        for cur_json_obj in f_reader:
            if cur_json_obj['context_length'] > min_len and cur_json_obj['context_length'] < max_len:
                positon_idx = translate_pos_ratio(cur_json_obj['mean_pos'], interval_number=interval_number)
                if cur_json_obj['id'] in id_score_dict.keys():
                    tmp_score = id_score_dict[cur_json_obj['id']]
                    positon_scores.setdefault(positon_idx,[]).append(tmp_score)
    
    results_dict = {}

    for key in sorted(positon_scores.keys()):
        avg_score = round(sum(positon_scores[key]) / float(len(positon_scores[key])),2)
        results_dict[key] = avg_score
        print (key, len(positon_scores[key]))
    results_dict['model_name'] = args.model_name
    results_dict['interval_number'] = interval_number
    results_dict['min_len'] = min_len
    results_dict['max_len'] = max_len

    with jsonlines.open("./eval_position_results/{}.jsonl".format(args.dataset), "a") as writer:
        writer.write(results_dict)