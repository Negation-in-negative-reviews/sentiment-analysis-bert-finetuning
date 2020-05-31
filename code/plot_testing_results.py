# python code/plot_testing_results.py --saves_dir=testing_pickle_saves --bargraph_savepath=testing_pickle_saves/testing_accuracies

import pickle
import argparse
import os
import pandas as pd
import plotting_code

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def read_and_process(args):
    names_map = {
        "yelp": "Yelp", 
        "imdb": "IMDB", 
        "tripadvisor": "Tripadvisor", 
        "automotive": "Automotive", 
        "luxury_beauty": "Luxury Beauty", 
        "cellphones_and_accessories": "Cellphones", 
        "sports_and_outdoors": "Sports", 
        "pet_supplies": "Pet Supplies"
    }
    plot_df = pd.DataFrame()
    for subdir in os.listdir(args.saves_dir):
        # print(subdir)
        if os.path.isdir(os.path.join(args.saves_dir, subdir)):
            pos_file = os.path.join(args.saves_dir, subdir, "pos_sents_test")        
            neg_file = os.path.join(args.saves_dir,subdir, "neg_sents_test")
            # print(neg_file)
            data_pos = pickle.load(open(pos_file, "rb"))
            data_neg = pickle.load(open(neg_file, "rb"))
            # print(data_neg)
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
                else:
                    data_df_plot = data_neg

                if args.correction_score == True:
                    correct_count = data_df_plot["correct_count"].to_list()[0]
                    total_count = data_df_plot["total_count"].to_list()[0]
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

                if args.correction_score == True:
                    accuracy = data_df_plot["accuracy"].to_list()[0]
                    true_rate = data_df_plot["correct_count_with_negation"].to_list()[0]/data_df_plot["total_negation_count"].to_list()[0]
                    false_rate = data_df_plot["incorrect_count_with_negation"].to_list()[0]/data_df_plot["total_negation_count"].to_list()[0]
                    acc = true_rate*accuracy + false_rate*(1-accuracy)
                else:
                    acc = data_df_plot["correct_count_with_negation"].to_list()[0]/data_df_plot["total_negation_count"].to_list()[0]

                plot_df = plot_df.append({
                    "name": names_map[dataset_name],
                    "category": "sentences with negation",
                    "accuracy": acc,
                }, ignore_index=True)

                if args.correction_score == True:
                    accuracy = data_df_plot["accuracy"].to_list()[0]
                    true_rate = data_df_plot["correct_count_with_pos_words"].to_list()[0]/data_df_plot["total_count_with_pos_words"].to_list()[0]
                    false_rate = data_df_plot["incorrect_count_with_pos_words"].to_list()[0]/data_df_plot["total_count_with_pos_words"].to_list()[0]
                    acc = true_rate*accuracy + false_rate*(1-accuracy)
                else:
                    acc = data_df_plot["correct_count_with_pos_words"].to_list()[0]/data_df_plot["total_count_with_pos_words"].to_list()[0]
                
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
        colors = [(173/255, 160/255, 160/255),(67/255, 144/255, 188/255), (2/255, 72/255, 110/255)]*2

    # print(plot_df)

    plot_df_amz = plot_df[plot_df["name"].isin(amazon_names)]
    plot_df_non_amz = plot_df[~plot_df["name"].isin(amazon_names)]

    if args.label == 'positive':
        y_axis_name = "% of pos. sentences"
    else:
        y_axis_name = "% of neg. sentences"

    ylim_top = plot_df_amz.max(axis=0)["accuracy"]
    ylim_top = 1.7*ylim_top
        
    if args.correction_score:
        save_path = args.bargraph_savepath+"_"+args.label+"_reviews_with_correction"
    else:
        save_path = args.bargraph_savepath+"_"+args.label+"_reviews"

    if args.type == "normal_plot":
        plotting_code.draw_grouped_barplot(plot_df_amz, colors, "name", "accuracy", 
        "review_category", save_path+"_amz", ylim_top=ylim_top, y_axis_name=y_axis_name)
    elif args.type == "negation_positive_plot":
        plotting_code.draw_grouped_barplot_three_subbars(plot_df_amz, colors, "name", "accuracy", 
        "category", save_path+"_amz", ylim_top=ylim_top, y_axis_name=y_axis_name)

    
    ylim_top = plot_df_non_amz.max(axis=0)["accuracy"]
    ylim_top = 1.4*ylim_top
    
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