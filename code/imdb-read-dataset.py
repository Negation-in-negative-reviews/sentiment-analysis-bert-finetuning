import os

IMDB_DATASET_PATH = "/data/madhu/imdb_dataset/aclImdb/train/"

def read_files(catgeory):
    files_dir = os.path.join(IMDB_DATASET_PATH, catgeory)
    
    onlyfiles = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))]
    out_dir = "/data/madhu/imdb_dataset/processed_data"

    f_out = open(os.path.join(out_dir, catgeory+"_samples_full"), "w")
    
    for file in onlyfiles:
        file_path = os.path.join(files_dir, file)
        with open(file_path, "r") as f:
            f_out.write(f.readline().strip("\n").replace("<br />", ". ")+"\n")

if __name__ == "__main__":
    read_files("pos")
    read_files("neg")


