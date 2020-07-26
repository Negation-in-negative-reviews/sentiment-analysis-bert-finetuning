#!/bin/sh/
nohup python code/get-sents.py --pos_file="/data/madhu/yelp/yelp_processed_data/review.0_5000samples" --neg_file="/data/madhu/yelp/yelp_processed_data/review.1_5000samples" &
nohup python code/get-sents.py --pos_file="/data/madhu/imdb_dataset/processed_data/pos_samples_full" --neg_file="/data/madhu/imdb_dataset/processed_data/neg_samples_full" &
