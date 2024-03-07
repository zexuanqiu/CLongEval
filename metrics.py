import jieba 
from rouge import Rouge 
import string
import re
from collections import Counter 
from post_processing_for_stacked_tasks import news_labeling_pattern_match, typo_detection_pattern_match


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# 1-1_long_story_qa; 1-2_long_conversation_memory
def qa_f1_zh_score(prediction, ground_truth):
    prediction = prediction.strip()

    # ad-hoc code for some llms that sometime self-question, e.g., chinese_alpaca_2_64k and qwen
    if "\n" in prediction:
        prediction = prediction.split("\n")[0]
    if "问题：" in prediction:
        pos_idx = prediction.find("问题：" )
        prediction = prediction[:pos_idx]
        
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

# 2-1_long_story_summarization
def rouge_zh_score(prediction, ground_truth):
    # https://pypi.org/project/rouge/
    if prediction == "":
        return 0.
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    rouge = Rouge() 
    scores_one_sample = rouge.get_scores(prediction, ground_truth)
    return scores_one_sample[0]["rouge-l"]["f"]

# 3-1_stacked_news_labeling
def news_labeling_zh_score(prediction, ground_truth):
    prediction_list = news_labeling_pattern_match(prediction.strip())
    if len(prediction_list) == 0:
        return 0.
    ground_truth_list = []
    pattern = r"新闻\s(\d+)[：](\w+)"
    for cur_news in ground_truth.split("\n"):
        match = re.search(pattern, cur_news)
        ground_truth_list.append("，".join([match.group(1), match.group(2)]))
    num_same =  len(set(prediction_list) & set(ground_truth_list))
    if num_same == 0:   return 0.
    res = 1.0 * num_same / len(ground_truth_list)
    return res


# 3-2_stacked_typo_detection
# Score for detecting typos: If the paragraph ID and the corresponding typo are found together, it will be considered correct.
def typo_detection_zh_score(prediction, ground_truth):
    prediction_list = typo_detection_pattern_match(prediction.strip())
    if len(prediction_list) == 0:
        return 0.
    # prediction_list = prediction.strip().split("\n")
    prediction_list = [("").join(i.split('，')[0:2]) for i in prediction_list]
    ground_truth_list = ground_truth.strip().split("\n")
    ground_truth_list = [("").join(i.split('，')[0:2]) for i in ground_truth_list]
    num_same =  len(set(prediction_list) & set(ground_truth_list))
    if num_same == 0:   return 0.
    res = 1.0 * num_same / len(ground_truth_list)
    return res



def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# 4-1_key_retrieval
def retrieval_zh_edit_score(prediction, ground_truth):
    if "键：" in prediction: # qwen
        pos_idx = prediction.find("键：" )
        prediction = prediction[:pos_idx]
    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)
    prediction = remove_punc(prediction)
    ground_truth = remove_punc(ground_truth)
    edit_distance = levenshtein_distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))
    score = (1. - float(edit_distance) /max_len)
    return score 

# 4-2_table_querying
def table_querying_match_score(prediction, ground_truth):
    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)
    prediction = remove_punc(prediction)
    ground_truth = remove_punc(str(ground_truth))
    # if the ground truth is included in the prediction, the score is 1, otherwise it is 0
    score = 1.0 if ground_truth in prediction else 0.0 # 
    return score 
