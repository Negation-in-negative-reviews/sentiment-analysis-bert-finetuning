# Command to run this file
# python code/plot_testing_results.py --saves_dir=testing_pickle_saves \
# --bargraph_savepath=testing_pickle_saves/testing_accuracies

import pickle
import argparse
import os
import pandas as pd
import plotting_code
from pathlib import Path

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

REVIEW_TESTING_DIR = "review_model_review_testing_outputs"

def read_and_process(args):
    names_map = {
        "yelp": "Yelp", 
        "imdb": "IMDB", 
        "tripadvisor": "Tripadvisor", 
        "cellphones_and_accessories": "Cellphones", 
        "pet_supplies": "Pet Supplies",
        "automotive": "Automotive", 
        "luxury_beauty": "Luxury Beauty", 
        "sports_and_outdoors": "Sports", 
    }
    plot_df = pd.DataFrame()
    
    for d in list(names_map.keys()):        
        if True:
            pos_file = os.path.join(args.saves_dir, d, "pos_sents_test")        
            neg_file = os.path.join(args.saves_dir, d, "neg_sents_test")
            
            data_pos = pickle.load(open(pos_file, "rb"))
            data_neg = pickle.load(open(neg_file, "rb"))

            pos_file_review_testing = os.path.join(REVIEW_TESTING_DIR, d, "pos_reviews_test")        
            neg_file_review_testing = os.path.join(REVIEW_TESTING_DIR, d, "neg_reviews_test")
            
            data_pos_review_testing = pickle.load(open(pos_file_review_testing, "rb"))
            data_neg_review_testing = pickle.load(open(neg_file_review_testing, "rb"))
            
            dataset_name = data_pos["name"].to_list()[0]
            if args.type == "normal_plot":
                plot_df = plot_df.append({
                    "name": names_map[dataset_name],
                    "review_category": "positive",
                    "accuracy": data_pos["accuracy"].to_list()[0],
                }, ignore_index=True)
                
                plot_df = plot_df.append({
                    "name": names_map[dataset_name],
                    "review_category": "negative",
                    "accuracy": data_neg["accuracy"].to_list()[0],
                }, ignore_index=True)
            elif args.type == "negation_positive_plot":
                if args.label == "positive":
                    data_df_plot = data_pos
                    data_df_plot_review_testing = data_pos_review_testing
                else:
                    data_df_plot = data_neg
                    data_df_plot_review_testing = data_neg_review_testing

                if args.correction_score == True:
                    correct_count = data_df_plot_review_testing["correct_count"].to_list()[0]
                    total_count = data_df_plot_review_testing["total_count"].to_list()[0]
                    accuracy = data_df_plot["accuracy"].to_list()[0]
                    true_rate = correct_count/total_count
                    false_rate = (total_count-correct_count)/total_count
                    acc = true_rate*accuracy + false_rate*(1-accuracy)
                else:                    
                    acc = data_df_plot["correct_count"].to_list()[0]/data_df_plot["total_count"].to_list()[0]

                plot_df = plot_df.append({
                    "name": names_map[dataset_name],
                    "category": "all sentences",
                    "accuracy": acc,
                }, ignore_index=True)

                acc = data_df_plot["correct_count_with_negation"].to_list()[0]/data_df_plot["total_negation_count"].to_list()[0]
                if args.correction_score == True:                    
                    true_rate = data_df_plot_review_testing["correct_count_with_negation"].to_list()[0]/ \
                        data_df_plot_review_testing["total_negation_count"].to_list()[0]
                    false_rate = data_df_plot_review_testing["incorrect_count_with_negation"].to_list()[0]/ \
                        data_df_plot_review_testing["total_negation_count"].to_list()[0]
                    acc = true_rate*acc + false_rate*(1-acc)
                    

                plot_df = plot_df.append({
                    "name": names_map[dataset_name],
                    "category": "sentences with negation",
                    "accuracy": acc,
                }, ignore_index=True)

                acc = data_df_plot["correct_count_with_pos_words"].to_list()[0]/data_df_plot["total_count_with_pos_words"].to_list()[0]
                if args.correction_score == True:                    
                    true_rate = data_df_plot_review_testing["correct_count_with_pos_words"].to_list()[0]/ \
                        data_df_plot_review_testing["total_count_with_pos_words"].to_list()[0]
                    false_rate = data_df_plot_review_testing["incorrect_count_with_pos_words"].to_list()[0]/ \
                        data_df_plot_review_testing["total_count_with_pos_words"].to_list()[0]
                    acc = true_rate*acc + false_rate*(1-acc)
                    
                
                plot_df = plot_df.append({
                    "name": names_map[dataset_name],
                    "category": "sentences with positive lexicons",
                    "accuracy": acc,
                }, ignore_index=True)    

    
    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']
    amazon_names = [val for val in amazon_names]   
    if args.type == "normal_plot":
        colors = [(84/255, 141/255, 255/255),  (84/255, 141/255, 255/255)]*2
    elif args.type == "negation_positive_plot":
        colors = [(183/255, 183/255, 183/255),(67/255, 144/255, 188/255), (2/255, 72/255, 110/255)]*2
    
    Path(os.path.join(args.saves_dir, "df_outputs")).mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(os.path.join(args.saves_dir, "df_outputs",
        'df_outputs_correction_'+str(args.correction_score)+'_'+args.type+'_'+args.label+'_reviews.csv') )


    plot_df_amz = plot_df[plot_df["name"].isin(amazon_names)]
    plot_df_non_amz = plot_df[~plot_df["name"].isin(amazon_names)]

    if args.label == 'positive':
        y_axis_name = "\% of pos. sentences"
    else:
        y_axis_name = "\% of neg. sentences"

    ylim_top = plot_df_amz.max(axis=0)["accuracy"]
    ylim_top = 1.7*ylim_top

    Path(os.path.dirname(args.bargraph_savepath)).mkdir(parents=True, exist_ok=True)
        
    if args.correction_score:
        save_path = args.bargraph_savepath+"_"+args.label+"_reviews_with_correction"
    else:
        save_path = args.bargraph_savepath+"_"+args.label+"_reviews"

    if args.type == "normal_plot":
        plotting_code.draw_grouped_barplot(plot_df_amz, colors, "name", "accuracy", 
        "review_category", save_path+"_amz", ylim_top=ylim_top, y_axis_name=y_axis_name, amazon_data_flag=True)
    elif args.type == "negation_positive_plot":
        plotting_code.draw_grouped_barplot_three_subbars(plot_df_amz, colors, "name", "accuracy", 
        "category", save_path+"_amz", ylim_top=ylim_top, y_axis_name=y_axis_name, amazon_data_flag=True)

    
    ylim_top = plot_df_non_amz.max(axis=0)["accuracy"]
    ylim_top = 1.7*ylim_top
    
    if args.type == "normal_plot":
        plotting_code.draw_grouped_barplot(plot_df_non_amz, colors, "name", "accuracy", 
        "review_category", save_path+"_non_amz", ylim_top=ylim_top, y_axis_name=y_axis_name)
    elif args.type == "negation_positive_plot":
        plotting_code.draw_grouped_barplot_three_subbars(plot_df_non_amz, colors, "name", "accuracy", 
        "category", save_path+"_non_amz", ylim_top=ylim_top, y_axis_name=y_axis_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--saves_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--bargraph_savepath",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--type",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--label",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--correction_score",
                        action='store_true',
                        help="")

    args = parser.parse_args()

    read_and_process(args)