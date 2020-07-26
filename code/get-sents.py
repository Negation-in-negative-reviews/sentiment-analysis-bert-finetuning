import sys
import spacy
import os
import argparse

nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def write_sents_to_file(in_file, out_file):
    f_out = open(out_file, "w")
    count = 0
    with open(in_file, "r") as f_in:
        for review in f_in.readlines():            
            doc = nlp(review)
            sentences = list(doc.sents)
            for sent in sentences:
                tokens = tokenizer(sent.string.strip())
                if len(tokens) >= 5:
                    f_out.write(sent.string.strip().strip("\n")+"\n")                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--neg_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")    
    
    args = parser.parse_args()     

    pos_out_file = args.pos_file+"_sents"
    neg_out_file = args.neg_file+"_sents"

    print("pos_out_file: ", pos_out_file)
    print("neg_out_file: ", neg_out_file)

    write_sents_to_file(args.pos_file, pos_out_file)
    write_sents_to_file(args.neg_file, neg_out_file)


    

    
