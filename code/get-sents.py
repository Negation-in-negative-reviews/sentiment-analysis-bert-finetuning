import sys
import spacy
import os
# nlp = English()

nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def write_sents_to_file(in_file, out_file):
    f_out = open(out_file, "w")
    count = 0
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    with open(in_file, "r") as f_in:
        for review in f_in.readlines():            
            doc = nlp(review)
            sentences = list(doc.sents)
            for sent in sentences:
                tokens = tokenizer(sent.string.strip())
                if len(tokens) >= 5:
                    f_out.write(sent.string.strip().strip("\n")+"\n")                

if __name__ == "__main__":

    # Yelp dataset
    pos_file = "/data/madhu/yelp/yelp_processed_data/review.0_5000samples"
    neg_file = "/data/madhu/yelp/yelp_processed_data/review.1_5000samples"

    # Imdb dataset
    # pos_file = "/data/madhu/imdb_dataset/processed_data/pos_samples_full"
    # neg_file = "/data/madhu/imdb_dataset/processed_data/neg_samples_full"

    pos_out_file = pos_file+"_sents"
    neg_out_file = neg_file+"_sents"

    print(pos_out_file)
    print(neg_out_file)

    write_sents_to_file(pos_file, pos_out_file)
    write_sents_to_file(neg_file, neg_out_file)


    

    
