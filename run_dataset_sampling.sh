#!/bin/sh/
nohup python code/dataset-sampling.py --pos_file="/data/madhu/imdb_dataset/processed_data/pos_samples_full_sents" -neg_file="/data/madhu/imdb_dataset/processed_data/neg_samples_full_sents" --n_samples=10000 &
nohup python code/dataset-sampling.py --pos_file="/data/madhu/yelp/yelp_processed_data/review.1" -neg_file="/data/madhu/yelp/yelp_processed_data/review.0" --n_samples=6000 &
nohup python code/dataset-sampling.py --pos_file="/data/madhu/yelp/shen_et_al_data/sentiment.train.1" -neg_file="/data/madhu/yelp/shen_et_al_data/sentiment.train.0" --n_samples=5000 &
