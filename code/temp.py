
# samples = []
# with open("/data/madhu/yelp/yelp_processed_data/review.1_6000samples", "r") as f:
#     for line in f:
#         samples.append(line.strip().strip("\n"))
    
# with open("/data/madhu/yelp/yelp_processed_data/review.1_5000samples", "w") as fout:
#     for s in samples[:5000]:
#         fout.write(s+"\n")

# with open("/data/madhu/yelp/yelp_processed_data/review.1_1000samples_test", "w") as fout:
#     for s in samples[5000:]:
#         fout.write(s+"\n")

# import numpy as np

# seed_val = 23
# np.random.seed(seed_val)
# reviews = []
# with open("/data/madhu/yelp/shen_et_al_data/sentiment.test.1", "r") as f:
#     for line in f:
#         reviews.append(line.strip().strip("\n"))
    
# indices = np.random.choice(np.arange(len(reviews)), size=1000)

# samples = [reviews[idx] for idx in indices]

# with open("/data/madhu/yelp/shen_et_al_data/sentiment.test.1_1000samples", "w") as fout:
#     for s in samples:
#         fout.write(s+"\n")

# filename = "/data/madhu/yelp/yelp_processed_data/review.1_test"
# with open(filename, "r") as fin:
#     reviews = fin.readlines()
#     reviews = [rev.lower().strip("\n") for rev in reviews]

#     train_size = int(2.0*len(reviews)/3)
#     train_reviews = reviews[:train_size]
#     test_reviews = reviews[train_size+1:]
    
#     with open(filename+"_train", "w") as fout:
#         for rev in train_reviews:
#             fout.write(rev.strip("\n")+"\n")

#     with open(filename+"_test", "w") as fout:
#         for rev in test_reviews:
#             fout.write(rev.strip("\n")+"\n")


import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")
filename = "/data/madhu/yelp/yelp_processed_data/review.1_test"
fout = open(filename+"_50k_sents", "w")
all_sents = []
with open(filename, "r") as fin:
    for rev in fin.readlines():
        rev = rev.strip("\n")
        doc = nlp(rev)
        for sent in doc.sents:
            # fout.write(sent.text.strip()+"\n")
            all_sents.append(sent.text)
    indices = np.random.choice(np.arange(len(all_sents)), size=50000)
    selected_sents = [all_sents[idx] for idx in indices]
    for sent in selected_sents:
        fout.write(sent.strip()+"\n")