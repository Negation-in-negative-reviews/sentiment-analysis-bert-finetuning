import numpy as np

def read_file(filename):

    data = []
    with open(filename, "r") as fin:
        for line in fin:
            data.append(line.strip("\n"))
        return data

def uniform_sampling(data, n_samples):
    sampled_data = []
    indices = np.random.choice(np.arange(len(data)), size=n_samples)
    sampled_data = [data[idx] for idx in indices]
    return sampled_data

def write_file(filename, data):

    with open(filename, "w") as fout:
        for d in data:
            fout.write(d+"\n")


if __name__=="__main__":
    seed_val = 23
    np.random.seed(seed_val)

    filenames = [
        "/data/madhu/imdb_dataset/processed_data/neg_test_reviews_same_review_sents",
        "/data/madhu/imdb_dataset/processed_data/pos_test_reviews_same_review_sents"
    ]
    out_filenames = [
        "/data/madhu/imdb_dataset/processed_data/50_samples_neg",
        "/data/madhu/imdb_dataset/processed_data/50_samples_pos"
    ]
    n_samples = int(5e1)

    for fin, fout in zip(filenames, out_filenames):
        data = read_file(fin)
        sampled_data = uniform_sampling(data, n_samples)
        write_file(fout, sampled_data)