for cur_dataset in {long_story_qa,long_conversation_memory,long_story_summarization,stacked_news_labeling,stacked_typo_detection,key_passage_retrieval,table_querying}
do
    python eval.py --model_name gpt4-turbo-128k --datasets $cur_dataset
done    
# python eval.py --model_name moonshot-v1 --datasets stacked_typo_detection