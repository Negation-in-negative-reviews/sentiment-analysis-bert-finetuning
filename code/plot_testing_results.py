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
    plot_df = pd.DataFrame(columns=["name", "review_category", "accuracy"])
    for subdir in os.listdir(args.saves_dir):
        print(subdir)
        if os.path.isdir(os.path.join(args.saves_dir, subdir)):
            pos_file = os.path.join(args.saves_dir, subdir, "pos_sents_test")        
            neg_file = os.path.join(args.saves_dir,subdir, "neg_sents_test")

            data = pickle.load(open(pos_file, "rb"))
            dataset_name = data["name"].to_list()[0]
            plot_df = plot_df.append({
                "name": names_map[dataset_name],
                "review_category": "positive",
                "accuracy": data["accuracy"].to_list()[0],
            }, ignore_index=True)

            data = pickle.load(open(neg_file, "rb"))
            plot_df = plot_df.append({
                "name": names_map[dataset_name],
                "review_category": "negative",
                "accuracy": data["accuracy"].to_list()[0],
            }, ignore_index=True)

    
    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']
    amazon_names = [val for val in amazon_names]   

    plot_df_amz = plot_df[plot_df["name"].isin(amazon_names)]
    plot_df_non_amz = plot_df[~plot_df["name"].isin(amazon_names)]

    ylim_top = plot_df_amz.max(axis=0)["accuracy"]
    ylim_top = 1.4*ylim_top

    plotting_code.draw_grouped_barplot(plot_df_amz, "name", "accuracy", 
        "review_category", args.bargraph_savepath+"_amz", ylim_top=ylim_top)

    ylim_top = plot_df_non_amz.max(axis=0)["accuracy"]
    ylim_top = 1.4*ylim_top

    plotting_code.draw_grouped_barplot(plot_df_non_amz, "name", "accuracy", 
        "review_category", args.bargraph_savepath+"_non_amz", ylim_top=ylim_top)

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

    args = parser.parse_args()

    read_and_process(args)