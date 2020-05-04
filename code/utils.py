import torch
import os
import sys
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
from datetime import datetime
import logging
from pathlib import Path

def get_filename(time: int, util_name:str =""):   
    # print(args)    
    filename = str(time.strftime('%b-%d-%Y_%H-%M-%S'))
    if util_name != "":
        filename = util_name+"_"+filename
    return filename


# If there's a GPU available...
def get_device(device_no: int):
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda:"+str(device_no))

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    return device

def read_samples(filename0: str, filename1: str, seed_val:int, n_samples:int = None):
    yelp_reviews_0 = []
    yelp_reviews_1 = []
    with open(filename0, "r") as f:
        yelp_reviews_0 = f.readlines()
        
    with open(filename1, "r") as f:
        yelp_reviews_1 = f.readlines()

    seed_val = 23
    np.random.seed(seed_val)

    reviews = []
    labels = []
    if n_samples != None:
        indices = np.random.choice(np.arange(len(yelp_reviews_0)), size=n_samples)
        yelp_reviews_0_new = [yelp_reviews_0[idx] for idx in indices]

        indices = np.random.choice(np.arange(len(yelp_reviews_1)), size=n_samples)
        yelp_reviews_1_new = [yelp_reviews_1[idx] for idx in indices]
        
        reviews = yelp_reviews_0_new+yelp_reviews_1_new
        labels = [0]*len(yelp_reviews_0_new) + [1]*len(yelp_reviews_1_new)
    else:
        reviews = yelp_reviews_0+yelp_reviews_1
        labels = [0]*len(yelp_reviews_0) + [1]*len(yelp_reviews_1)
    reviews = [rev.lower() for rev in reviews]
    return reviews, labels

