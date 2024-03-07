import jsonlines 
import argparse 
import re 

# By observing the output of GPT4-Turbo-128K, we summarize the common output patterns

def news_labeling_pattern_match(pred_answer:str):
    res_list = []
    # case1     新闻1，类别名：家居
    # case2     新闻1，类别名1：时政
    # case3     新闻1，教育     新闻 1：教育 (desired format)   
    pattern1 = r"新闻(\d+)，类别名：(\w+)"
    pattern2 = r"新闻(\d+)，类别名\d+：(\w+)"
    pattern3 = r"新闻\s?(\d+)[，：](\w+)"
    patterns = [pattern1, pattern2, pattern3]
    lines_list = pred_answer.split("\n")
    for cur_line in lines_list:
        for cur_pattern in patterns:
            match = re.search(cur_pattern, cur_line)
            if match:
                news_id = match.group(1)
                category_id = match.group(2)
                res_list.append("，".join([news_id, category_id]))
                break
    return res_list
    


def typo_detection_pattern_match(pred_answer:str):
    typo_list = pred_answer.split('\n')
    res_list = [] 

    # case1     段落ID：1，错别字1，姙，正确字1，娠
    # case2     段落ID：1，错别字，姙，正确字，娠
    # case3     段落ID：5，错别字：屎，正确字：使
    # case4     段落ID：1，错别字1，忎，忍
    # case5     段落ID：0，齥，饿
    # case6     0，齥，饿
    pattern1 = r"段落ID：(\d+)[，,]错别字\d+，(\w+)[，,]正确字\d+，(\w+)"
    pattern2 = r"段落ID：(\d+)[，,]错别字，(\w+)[，,]正确字，(\w+)"
    pattern3 = r"段落ID：(\d+)[，,]错别字：(\w+)[，,]正确字：(\w+)"
    pattern4 = r"段落ID：(\d+)[，,]错别字\d+，(\w+)[，,](\w+)"
    pattern5 = r"段落ID：(\d+)[，,](\w+)[，,](\w+)"
    pattern6 = r"(\d+)[，,](\w+)[，,](\w+)"
    patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6]
    for cur_typo in typo_list:
        left_parenthesis = cur_typo.find("（") 
        if left_parenthesis != -1:
            cur_typo = cur_typo[:left_parenthesis]
        for pattern in patterns:
            match = re.search(pattern, cur_typo)
            if match:
                id_number = match.group(1)
                wrong_word = match.group(2)
                correct_word = match.group(3)
                # print("，".join([id_number, wrong_word, correct_word]))
                res_list.append("，".join([id_number, wrong_word, correct_word]))
                break

    typo_line_list = pred_answer.split("\n\n")
    if len(typo_line_list) == 1: 
        # print("exit")
        return res_list
    # print("段落")

    # case7:
    # 段落ID：3
    # 错别字：灾
    # 正确字：站

    # case8
    # 段落ID：0
    # 错别字：毧，正确字：绒

    alternative_res_list = [] 
    pattern7 = r"段落ID：(\d+)\n错别字：(\w)\n正确字：(\w)"
    pattern8 = r"段落ID：(\d+)\n错别字：(\w)，正确字：(\w)"
    alternative_patterns = [pattern7, pattern8]
    for pattern in alternative_patterns:
        match = re.search(pattern, cur_typo)
        if match:
            id_number = match.group(1)
            wrong_word = match.group(2)
            correct_word = match.group(3)
            alternative_res_list.append("，".join([id_number, wrong_word, correct_word]))
            break
    final_res_list = list(set(res_list) & set(alternative_res_list))
    return final_res_list