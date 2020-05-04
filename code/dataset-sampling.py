import numpy as np

def get_random_samples(n_samples:int, filename: str, rand_flag:bool):

    samples = []
    with open(filename, "r") as fin:
        for line in fin:
            samples.append(line.strip().strip("\n"))
    
    indices = np.arange(len(samples))

    result = []
    if rand_flag:
        indices = np.random.choice(len(samples), size=n_samples)
        for idx in indices:
            result.append(samples[idx])
    else:
        result = samples[:n_samples]

    return result

def write_to_file(filename: str, data: list):

    with open(filename, "w") as fout:
        for line in data:
            fout.write(line+"\n")


if __name__ == "__main__":
    seed_val = 23
    np.random.seed(seed_val)
    
    # n_samples = 10000
    # file0 = "/data/madhu/imdb_dataset/processed_data/neg_samples_full_sents"
    # file1 = "/data/madhu/imdb_dataset/processed_data/pos_samples_full_sents"

    n_samples = int(6e3)
    file0 = "/data/madhu/yelp/yelp_processed_data/review.0"
    file1 = "/data/madhu/yelp/yelp_processed_data/review.1"

    # n_samples = int(5e3)
    # file0 = "/data/madhu/yelp/shen_et_al_data/sentiment.train.0"
    # file1 = "/data/madhu/yelp/shen_et_al_data/sentiment.train.1"

    fileout0 = file0+"_"+str(n_samples)+"samples"
    fileout1 = file1+"_"+str(n_samples)+"samples"

    # fileout0 = file0+"_"+str(n_samples)+"samples_traintest"
    # fileout1 = file1+"_"+str(n_samples)+"samples_traintest"

    samples0 = get_random_samples(n_samples, file0, True)
    write_to_file(fileout0, samples0)

    samples1 = get_random_samples(n_samples, file1, True)
    write_to_file(fileout1, samples1)

