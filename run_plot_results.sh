#!/bin/sh
for SENT in "positive" "negative"
do 
    # python code/plot_testing_results.py --saves_dir=review_model_sentence_testing_outputs/ --bargraph_savepath=review_model_sentence_testing_outputs/plots/reviews_normal --type="normal_plot" --label="$SENT"

    # python code/plot_testing_results.py --saves_dir=review_model_sentence_testing_outputs/ --bargraph_savepath=review_model_sentence_testing_outputs/plots/reviews_normal --type="normal_plot" --correction_score --label="$SENT"
    
    python code/plot_testing_results.py --saves_dir=review_model_sentence_testing_outputs/ --bargraph_savepath=review_model_sentence_testing_outputs/plots/reviews_with_negation --type="negation_positive_plot" --label="$SENT"

    python code/plot_testing_results.py --saves_dir=review_model_sentence_testing_outputs/ --bargraph_savepath=review_model_sentence_testing_outputs/plots/reviews_with_negation --type="negation_positive_plot" --correction_score --label="$SENT"

done
