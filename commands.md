```
nohup python code/testing.py \
--input_file=/data/madhu/yelp_shen_et_al_aws/processed_files_with_bert_with_best_head/sentiment.test.1_out \
--label="negative" 
--truncation="head-and-tail" \
--model_path=saves/yelp_shen_et_al/model_1epochs \
&> nohup_yelp_shen_et_al_testing_pos_to_neg.out
```