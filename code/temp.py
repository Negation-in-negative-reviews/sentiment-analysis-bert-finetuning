
samples = []
with open("/data/madhu/yelp/yelp_processed_data/review.0_6000samples_traintest", "r") as f:
    for line in f:
        samples.append(line.strip().strip("\n"))
    
with open("/data/madhu/yelp/yelp_processed_data/review.0_5000samples", "w") as fout:
    for s in samples[:5000]:
        fout.write(s+"\n")

with open("/data/madhu/yelp/yelp_processed_data/review.0_1000samples_test", "w") as fout:
    for s in samples[5000:]:
        fout.write(s+"\n")

