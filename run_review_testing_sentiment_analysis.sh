#!/bin/sh

nohup python code/testing.py --input_file=/data/madhu/yelp/yelp_processed_data_old/review.0_test --label="negative" --n_samples=5000 --name="yelp" --model_path=/data/madhu/models/bert-finetuning/saves/yelp-reviews/model_2epochs --device_no=1 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/yelp_neg.out &

nohup python code/testing.py --input_file=/data/madhu/yelp/yelp_processed_data_old/review.1_test  --label="positive" --n_samples=5000 --name="yelp" --model_path=/data/madhu/models/bert-finetuning/saves/yelp-reviews/model_2epochs --device_no=2 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/yelp_pos.out &



nohup python code/testing.py --input_file=/data/madhu/imdb_dataset/processed_data/neg_reviews_test --label="negative" --n_samples=5000 --name="imdb" --model_path=/data/madhu/models/bert-finetuning/saves/imdb/model_2epochs --device_no=3 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/imdb_neg.out &

nohup python code/testing.py --input_file=/data/madhu/imdb_dataset/processed_data/pos_reviews_test --label="positive" --n_samples=5000 --model_path=/data/madhu/models/bert-finetuning/saves/imdb/model_2epochs --device_no=2 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/  --name="imdb"  &> review_model_review_testing_outputs/imdb_pos.out &




nohup python code/testing.py --input_file=/data/madhu/tripadvisor/processed_data/neg_reviews_test --n_samples=5000 --label="negative"  --name="tripadvisor" --model_path=/data/madhu/models/bert-finetuning/saves/tripadvisor/model_2epochs --device_no=2 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/tripadvisor_neg.out &

nohup python code/testing.py --input_file=/data/madhu/tripadvisor/processed_data/pos_reviews_test --n_samples=5000 --label="positive" --name="tripadvisor"  --model_path=/data/madhu/models/bert-finetuning/saves/tripadvisor/model_2epochs --device_no=1 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/tripadvisor_pos.out &



nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/automotive/neg_reviews_test --name="automotive" --n_samples=5000 --label="negative" --model_path=/data/madhu/models/bert-finetuning/saves/automotive/model_2epochs --device_no=2 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/automotive_neg.out &

nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/automotive/pos_reviews_test --name="automotive" --n_samples=5000 --label="positive" --model_path=/data/madhu/models/bert-finetuning/saves/automotive/model_2epochs --device_no=1 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/automotive_pos.out &




nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/luxury_beauty/neg_reviews_test --name="luxury_beauty" --n_samples=5000 --label="negative" --model_path=/data/madhu/models/bert-finetuning/saves/luxury_beauty/model_2epochs --device_no=2 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/luxury_beauty_neg.out &

nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/luxury_beauty/pos_reviews_test --name="luxury_beauty" --n_samples=5000 --label="positive" --model_path=/data/madhu/models/bert-finetuning/saves/luxury_beauty/model_2epochs --device_no=1 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/luxury_beauty_pos.out &



nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/cellphones_and_accessories/neg_reviews_test  --name="cellphones_and_accessories" --n_samples=5000 --label="negative" --model_path=/data/madhu/models/bert-finetuning/saves/cellphones_and_accessories/model_2epochs --device_no=2 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/cellphones_and_accessories_neg.out &

nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/cellphones_and_accessories/pos_reviews_test  --name="cellphones_and_accessories" --n_samples=5000 --label="positive" --model_path=/data/madhu/models/bert-finetuning/saves/cellphones_and_accessories/model_2epochs --device_no=1 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/cellphones_and_accessories_pos.out &



nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/pet_supplies/neg_reviews_test --n_samples=5000 --name="pet_supplies" --label="negative" --model_path=/data/madhu/models/bert-finetuning/saves/pet_supplies/model_2epochs --device_no=2 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/pet_supplies_neg.out &

nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/pet_supplies/pos_reviews_test --n_samples=5000 --name="pet_supplies" --label="positive" --model_path=/data/madhu/models/bert-finetuning/saves/pet_supplies/model_2epochs --device_no=1 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/pet_supplies_pos.out &




nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/sports_and_outdoors/neg_reviews_test --n_samples=5000 --name="sports_and_outdoors" --label="negative" --model_path=/data/madhu/models/bert-finetuning/saves/sports_and_outdoors/model_2epochs --device_no=2 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/sports_and_outdoors_neg.out &

nohup python code/testing.py --input_file=/data/madhu/amazon-reviews-2018/processed_data/sports_and_outdoors/pos_reviews_test --n_samples=5000 --name="sports_and_outdoors" --label="positive" --model_path=/data/madhu/models/bert-finetuning/saves/sports_and_outdoors/model_2epochs --device_no=1 --truncation='head-and-tail' --saves_dir=./review_model_review_testing_outputs/ &> review_model_review_testing_outputs/sports_and_outdoors_pos.out &