for cur_dataset in {long_story_qa,long_conversation_memory,key_passage_retrieval,table_querying}
do 
    python eval_with_position.py --dataset $cur_dataset --model_name internlm2-7b-200k --interval_number 6 --min_len 10000 --max_len 60000
done 

for cur_dataset in {long_story_qa,long_conversation_memory,key_passage_retrieval,table_querying}
do 
    python eval_with_position.py --dataset $cur_dataset --model_name internlm2-20b-200k --interval_number 6 --min_len 10000 --max_len 60000
done 

for cur_dataset in {long_story_qa,long_conversation_memory,key_passage_retrieval,table_querying}
do 
    python eval_with_position.py --dataset $cur_dataset --model_name chinese-llama2-7b-64k --interval_number 6 --min_len 10000 --max_len 60000
done 

for cur_dataset in {long_story_qa,long_conversation_memory,key_passage_retrieval,table_querying}
do 
    python eval_with_position.py --dataset $cur_dataset --model_name chinese-alpaca2-7b-64k --interval_number 6 --min_len 10000 --max_len 60000
done 

for cur_dataset in {long_story_qa,long_conversation_memory,key_passage_retrieval,table_querying}
do 
    python eval_with_position.py --dataset $cur_dataset --model_name kimichat-128k --interval_number 6 --min_len 10000 --max_len 60000
done 

for cur_dataset in {long_story_qa,long_conversation_memory,key_passage_retrieval,table_querying}
do 
    python eval_with_position.py --dataset $cur_dataset --model_name gpt4-turbo-128k --interval_number 6 --min_len 10000 --max_len 60000
done