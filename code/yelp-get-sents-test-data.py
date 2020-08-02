
import spacy
nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def read_file(filename):
    reviews = []
    with open(filename, "r") as fin:
        for line in fin:
            reviews.append(line.strip().strip("\n"))
    return reviews

def get_sents(reviews):
    selected_sents = []    
    for review in reviews:   
        doc = nlp(review)
        sentences = list(doc.sents)
        for sent in sentences:
            selected_sents.append(sent.string.strip().strip("\n")) 

    return selected_sents


def write_sents(sents, outfile):
    with open(outfile, "w") as fout:
        for s in sents:
            fout.write(s+"\n")

if __name__=="__main__":

    testfile_pos = "/data/madhu/yelp/bert-finetuning-dataset/review.1_1000samples_test"
    testfile_neg = "/data/madhu/yelp/bert-finetuning-dataset/review.0_1000samples_test"

    test_reviews_pos = read_file(testfile_pos)
    test_reviews_neg = read_file(testfile_neg)

    pos_sents = get_sents(test_reviews_pos)
    neg_sents = get_sents(test_reviews_neg)

    outfile_pos = "/data/madhu/yelp/bert-finetuning-dataset/review.1_test_sents"
    outfile_neg = "/data/madhu/yelp/bert-finetuning-dataset/review.0_test_sents"

    write_sents(pos_sents, outfile_pos)
    write_sents(neg_sents, outfile_neg)

