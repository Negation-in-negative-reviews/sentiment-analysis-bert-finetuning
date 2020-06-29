import os
import spacy
import numpy as np

def read_file(filename):
    reviews = []
    with open(filename, "r") as fin:
        for line in fin:
            reviews.append(line.strip().strip("\n"))
    return reviews

def write_random_samples(outfile, n_samples, reviews):
    indices = np.random.choice(np.arange(len(reviews)), size=n_samples)
    selected = [reviews[idx] for idx in indices]

    with open(outfile, "w") as fout:
        for rev in selected:
            fout.write(rev+"\n")
    return selected

nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def get_sents(reviews):
    selected_sents = []    
    for review in reviews:   
        doc = nlp(review)
        sentences = list(doc.sents)
        for sent in sentences:
            # tokens = tokenizer(sent.string.strip())
            # if len(tokens) >= 5:
            selected_sents.append(sent.string.strip().strip("\n")) 

    return selected_sents

def write_sents(sents, outfile):
    with open(outfile, "w") as fout:
        for s in sents:
            fout.write(s+"\n")

if __name__ == "__main__":
    seed_val = 23
    np.random.seed(seed_val)

    pos_file = "/data/madhu/yelp/yelp_processed_data/review.1"
    neg_file = "/data/madhu/yelp/yelp_processed_data/review.0"

    pos_reviews = read_file(pos_file)
    neg_reviews = read_file(neg_file)

    n_samples = 5000
    random_pos_reviews = write_random_samples(pos_file+"_"+str(n_samples)+"samples", n_samples, pos_reviews)
    random_neg_reviews = write_random_samples(neg_file+"_"+str(n_samples)+"samples", n_samples, neg_reviews)
    
    pos_sents = get_sents(pos_reviews)
    neg_sents = get_sents(neg_reviews)

    n_sents = int(1e4)

    pos_indices = np.random.choice(np.arange(len(pos_sents)), size=n_samples)
    neg_indices = np.random.choice(np.arange(len(neg_sents)), size=n_samples)

    pos_selected_sents = [pos_sents[idx] for idx in pos_indices]
    neg_selected_sents = [neg_sents[idx] for idx in neg_indices]

    out_dir = "/data/madhu/imdb_dataset/processed_data"

    write_sents(pos_selected_sents, os.path.join(out_dir,"pos_reviews_"+str(n_samples)+"sents"))
    write_sents(neg_selected_sents, os.path.join(out_dir,"neg_reviews_"+str(n_samples)+"sents"))






