import os
import spacy
import numpy as np

# IMDB_DATASET_PATH = "/data/madhu/imdb_dataset/aclImdb/train/"
IMDB_DATASET_PATH = "/data/madhu/imdb_dataset/aclImdb/test/"

def read_files(catgeory):
    files_dir = os.path.join(IMDB_DATASET_PATH, catgeory)
    
    onlyfiles = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))]
    out_dir = "/data/madhu/imdb_dataset/processed_data"

    # f_out = open(os.path.join(out_dir, catgeory+"_reviews_train"), "w")
    f_out = open(os.path.join(out_dir, catgeory+"_reviews_test"), "w")
    # f_out = open(os.path.join(out_dir, "temp_"+catgeory), "w")
    samples = []
    for file in onlyfiles:
        file_path = os.path.join(files_dir, file)
        with open(file_path, "r") as f:
            line = f.readline().strip("\n").replace("<br />", ". ")
            samples.append(line)
            f_out.write(line+"\n")
    return samples


nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def get_sents(reviews):
    selected_sents = []    
    for review in reviews:   
        doc = nlp(review)        
        for sent in doc.sents:
            tokens = tokenizer(sent.string.strip())
            if len(tokens) >= 5:
                selected_sents.append(sent.string.strip().strip("\n")) 

    return selected_sents

def write_sents(sents, out_file):
    with open(out_file, "w") as fout:
        for s in sents:
            fout.write(s.strip("\n")+"\n")

if __name__ == "__main__":
    seed_val = 23
    np.random.seed(seed_val)

    pos_samples = read_files("pos")
    neg_samples = read_files("neg")

    pos_sents = get_sents(pos_samples)
    neg_sents = get_sents(neg_samples)

    n_samples = int(1e4)

    pos_indices = np.random.choice(np.arange(len(pos_sents)), size=n_samples)
    neg_indices = np.random.choice(np.arange(len(neg_sents)), size=n_samples)

    pos_selected_sents = [pos_sents[idx] for idx in pos_indices]
    neg_selected_sents = [neg_sents[idx] for idx in neg_indices]

    out_dir = "/data/madhu/imdb_dataset/processed_data"

    # write_sents(pos_selected_sents, os.path.join(out_dir,"pos_reviews_"+str(n_samples)+"sents"))
    # write_sents(neg_selected_sents, os.path.join(out_dir,"neg_reviews_"+str(n_samples)+"sents"))

    write_sents(pos_selected_sents, os.path.join(out_dir,"pos_test_reviews_"+str(n_samples)+"sents"))
    write_sents(neg_selected_sents, os.path.join(out_dir,"neg_test_reviews_"+str(n_samples)+"sents"))

    n_samples = int(1e4)

    pos_indices = np.random.choice(np.arange(len(pos_samples)), size=n_samples)
    neg_indices = np.random.choice(np.arange(len(neg_samples)), size=n_samples)

    selected_pos_samples = [pos_samples[idx] for idx in pos_indices]
    selected_neg_samples = [neg_samples[idx] for idx in neg_indices]

    pos_sents_same_review = get_sents(selected_pos_samples)
    neg_sents_same_review = get_sents(selected_neg_samples)

    write_sents(pos_sents_same_review, os.path.join(out_dir,"pos_test_reviews_"+str(n_samples)+"sents_same_review"))
    write_sents(neg_sents_same_review, os.path.join(out_dir,"neg_test_reviews_"+str(n_samples)+"sents_same_review"))
