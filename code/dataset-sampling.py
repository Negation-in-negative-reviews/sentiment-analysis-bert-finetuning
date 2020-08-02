import numpy as np
import argparse

def get_random_samples(n_samples:int, filename: str, rand_flag:bool):    
    """Returns the specified number of samples using uniform random distribution, if rand_flag is enabled. 
    Otherwise, Returns the first n_samples samples

    Args:
        n_samples (int): number of samples to be selected
        filename (str): Name of the file to read data from
        rand_flag (bool): Flag representing whether we need to use ranodm distribution for sampling.

    Returns:
        result: Sampled data
    """    
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
    """Write the samples (data) to file

    Args:
        filename (str): filename of the file to write data to
        data (list): Data to be written to file
    """    
    with open(filename, "w") as fout:
        for line in data:
            fout.write(line+"\n")


if __name__ == "__main__":
    """Randomly samples specified number of samples from pos_file and neg_file and writes to two files
    """    
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
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,                        
                        help="")
    parser.add_argument("--n_samples",
                        default=None,
                        type=str,                        
                        help="")
    
    args = parser.parse_args()    
    np.random.seed(args.seed_val)

    fileout0 = args.neg_file+"_"+str(args.n_samples)+"samples"
    fileout1 = args.pos_file+"_"+str(args.n_samples)+"samples"

    samples0 = get_random_samples(args.n_samples, args.neg_file, True)
    write_to_file(fileout0, samples0)

    samples1 = get_random_samples(args.n_samples, args.pos_file, True)
    write_to_file(fileout1, samples1)

