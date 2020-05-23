* Testing
```
nohup python code/testing.py \
--label="negative" \
--truncation="head-and-tail" \
--model_path=saves/yelp_shen_et_al/model_1epochs \
--input_file=/data/chenhao/Sentiment-and-Style-Transfer/data/yelp/processed_files_with_bert_with_best_head/sentiment_test_1.txt_flipped_out \
&> nohup_yelp_shen_et_al_testing_pos_to_neg.out
```

* Training
```
nohup python code/training.py --pos_file=/data/chenhao/Sentiment-and-Style-Transfer/data/amazon/sentiment.train.1 --neg_file=/data/chenhao/Sentiment-and-Style-Transfer/data/amazon/sentiment.train.0 --truncation="head-and-tail" --n_samples=50000 --device_no=0 &> nohup_amazon_drg_training.out &
```