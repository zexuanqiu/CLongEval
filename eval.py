import os 
import argparse 
import jsonlines 
import sys 
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
    parser.add_argument("--datasets", type=str, default=None)
    return parser.parse_args(args)

def scorer(dataset_name, predictions, answers):
    assert len(predictions) == len(answers)
    total_score = 0.
    for prediction, ground_truth in zip(predictions, answers):
        score = dataset2metric[dataset_name](prediction=prediction, ground_truth=ground_truth)
        total_score += score 
    return round(100 * total_score / len(predictions), 5)
            
def get_prediction_answer(jsonl_file, model_name):
    predictions = [] 
    ground_truths = [] 
    with jsonlines.open(jsonl_file, 'r') as f_reader:
        for cur_json_obj in f_reader:
            cur_pred = cur_json_obj['response_{}'.format(model_name)]
            if "HTTP_ERROR" in cur_pred or "UNKNOW_ERROR" in cur_pred:
                # Due to network errors or other issues, there were a few instances where results could not be generated using gpt4-turbo-128k even after calling api 5 times. We have ignored the scoring calculations for these samples.
                continue
            if 'typo_detection' in jsonl_file and model_name != "gpt4-turbo-128k" and "single" not in jsonl_file: # ad_hoc code
                cur_ans = cur_json_obj['answer']
                cur_ans_list = cur_ans.split("，")
                cur_ans = "，".join([cur_ans_list[0], cur_ans_list[2], cur_ans_list[0]]) # pargraph_id, wrong_word, right_word
            else:
                cur_ans = cur_json_obj['answer']
            predictions.append(cur_pred)
            ground_truths.append(cur_ans)
    return predictions, ground_truths

if __name__ == "__main__":
    args = parse_args() 
    chosen_datasets = args.datasets.split("-")
    scores = []
    infer_model_dir = os.path.join('./inference_results', args.model_name)
    dataset_dirs = list(sorted(os.listdir(infer_model_dir)))
    if not os.path.exists("eval_results"):
        os.makedirs("eval_results")

    for cur_dataset_dir in dataset_dirs:
        if cur_dataset_dir in chosen_datasets:
            print(cur_dataset_dir)
            small_file  = os.path.join(infer_model_dir, cur_dataset_dir, "small.jsonl")
            if os.path.exists(small_file):
                small_predicitons, small_answers = get_prediction_answer(small_file, args.model_name)
                small_score = scorer(cur_dataset_dir, small_predicitons, small_answers)
            else:
                small_score = None
            
            medium_file = os.path.join(infer_model_dir, cur_dataset_dir, "medium.jsonl")
            if os.path.exists(medium_file):
                medium_predictions, medium_answers = get_prediction_answer(medium_file, args.model_name)
                medium_score = scorer(cur_dataset_dir, medium_predictions, medium_answers)
            else:
                medium_score = None 

            large_file = os.path.join(infer_model_dir, cur_dataset_dir, "large.jsonl")
            if os.path.exists(large_file):
                large_predictions, large_answers = get_prediction_answer(large_file, args.model_name)
                large_score = scorer(cur_dataset_dir, large_predictions, large_answers)
            else:
                large_score = None 

            if os.path.exists(small_file) and os.path.exists(medium_file) and os.path.exists(large_file):
                all_predictions = small_predicitons + medium_predictions + large_predictions
                all_answers = small_answers + medium_answers + large_answers
                all_score = scorer(cur_dataset_dir, all_predictions, all_answers)
            else:
                all_score = None
            
            cur_scores_dict = {"dataset": cur_dataset_dir, "small": small_score, "medium": medium_score, "large": large_score, "all": all_score, "metric": dataset2metric[cur_dataset_dir].__name__}
            scores.append(cur_scores_dict)

    with jsonlines.open("./eval_results/{}.jsonl".format(args.model_name), mode="a") as writer:
        for score_dict in scores:
            writer.write(score_dict)
    





    
