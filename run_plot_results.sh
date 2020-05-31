#!/bin/sh
for SENT in "positive" "negative"
do 
    python code/plot_testing_results.py --saves_dir=testing_pickle_saves --bargraph_savepath=testing_pickle_saves/reviews_normal --type="normal_plot" --label="$SENT"

    python code/plot_testing_results.py --saves_dir=testing_pickle_saves --bargraph_savepath=testing_pickle_saves/reviews_normal --type="normal_plot" --correction_score --label="$SENT"

    python code/plot_testing_results.py --saves_dir=testing_pickle_saves --bargraph_savepath=testing_pickle_saves/reviews_with_negation --type="negation_positive_plot" --correction_score --label="$SENT"

    python code/plot_testing_results.py --saves_dir=testing_pickle_saves --bargraph_savepath=testing_pickle_saves/reviews_with_negation --type="negation_positive_plot" --label="$SENT"
done
