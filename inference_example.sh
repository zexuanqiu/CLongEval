# InternLM2-7B-200k
# Small Set
for cur_dataset in {long_story_qa,long_conversation_memory,long_story_summarization,stacked_news_labeling,stacked_typo_detection,key_passage_retrieval,table_querying}
do
    python inference.py --model_name internlm2-7b-200k --size small.jsonl --dataset_name $cur_dataset --gpus "0" --gpu_memory_utilization 0.8 --tensor_parallel_size 1
done 

# Medium Set
for cur_dataset in {long_story_qa,long_conversation_memory,long_story_summarization,stacked_news_labeling,stacked_typo_detection,key_passage_retrieval,table_querying}
do
    python inference.py --model_name internlm2-7b-200k --size medium.jsonl --dataset_name $cur_dataset --gpus "0" --gpu_memory_utilization 0.8 --tensor_parallel_size 1
done 

# Large Set
for cur_dataset in {long_story_qa,long_conversation_memory,long_story_summarization,stacked_news_labeling,stacked_typo_detection,key_passage_retrieval,table_querying}
do
    python inference.py --model_name internlm2-7b-200k --size large.jsonl --dataset_name $cur_dataset --gpus "0" --gpu_memory_utilization 0.8 --tensor_parallel_size 1
done 
