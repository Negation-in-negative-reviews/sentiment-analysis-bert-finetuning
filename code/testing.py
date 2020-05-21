import torch
import os
import sys
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import random
from pathlib import Path
import util
import datetime
import logging
import pprint
import pandas as pd
import argparse

pp = pprint.PrettyPrinter(indent=4)
iprint = pp.pprint

def test(args: dict(), save_flag: bool, seed_val):
    
    device = util.get_device(device_no=args.device_no)   
    model = torch.load(args.model_path, map_location=device)
    # seed_val = 2346610

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    testfile = args.input_file
    true_label = args.label
    truncation = args.truncation
    n_samples = None
    if "n_samples" in args:
        n_samples = args.n_samples

    # saves_dir = "saves/"  
    # time = datetime.datetime.now()
    # saves_path = os.path.join(saves_dir, util.get_filename(time))
    # if save_flag:
    #     Path(saves_path).mkdir(parents=True, exist_ok=True)


    # log_path = os.path.join(saves_path, "testing.log")

    # logging.basicConfig(filename=log_path, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    # logger=logging.getLogger() 
    # logger.setLevel(logging.DEBUG)

    
    # Load the BERT tokenizer.
    # logger.info('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = 0
    reviews = []
    labels = []
    with open(testfile, "r") as fin:
        reviews = fin.readlines()
    
    reviews = [rev.lower() for rev in reviews]
    
    if n_samples == None:
        n_samples = len(reviews)

    indices = np.random.choice(np.arange(len(reviews)), size=n_samples)
    selected_reviews = [reviews[idx] for idx in indices]

    labels = [0 if true_label == "negative" else 1]*len(selected_reviews)
    # For every sentence...
    # for rev in selected_reviews:
    #     # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    #     input_ids = tokenizer.encode(rev, add_special_tokens=True)
    #     # Update the maximum sentence length.
    #     max_len = max(max_len, len(input_ids))
        
    # print('Max sentence length: ', max_len)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for rev in selected_reviews:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        input_id = tokenizer.encode(rev, add_special_tokens=True)
        if len(input_id) > 512:                        
            if truncation == "tail-only":
                # tail-only truncation
                input_id = [tokenizer.cls_token_id]+input_id[-511:]      
            elif truncation == "head-and-tail":
                # head-and-tail truncation       
                input_id = [tokenizer.cls_token_id]+input_id[1:129]+input_id[-382:]+[tokenizer.sep_token_id]
            else:
                # head-only truncation
                input_id = input_id[:511]+[tokenizer.sep_token_id]
                
            input_ids.append(torch.tensor(input_id).view(1,-1))
            attention_masks.append(torch.ones([1,len(input_id)], dtype=torch.long))
        else:
            encoded_dict = tokenizer.encode_plus(
                                rev,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 512,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Set the batch size.  
    batch_size = 8  

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    
    print('    DONE.')
    return predictions, true_labels

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--truncation",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--label",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--n_samples",
                        default=None,
                        type=int,
                        help="")
    parser.add_argument("--device_no",
                        default=2,
                        type=int,
                        help="")
    
    args = parser.parse_args()
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    seed_vals = [23]
    accuracies_df = pd.DataFrame(columns=['dataset', 'seed_val', 'accuracy', 'score'])

    iprint(f"Args: {args}")
    for seed_val in seed_vals:    
        iprint(f"seed val: {seed_val}")
        preds, true_labels = test(args, False, seed_val)
        # Combine the results across all batches. 
        flat_predictions = np.concatenate(preds, axis=0)

        # For each sample, pick the label (0 or 1) with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        # Combine the correct labels for each batch into a single list.
        flat_true_labels = np.concatenate(true_labels, axis=0)
        accuracy = 1.0*(np.sum(flat_predictions == flat_true_labels))/flat_predictions.shape[0]        
        score = 0

        if args.label == "positive":
            correct_count = np.sum(flat_predictions == flat_true_labels)
            total_count = flat_predictions.shape[0]
            TPR = 1.0*correct_count/total_count
            FNR = 1.0*(total_count-correct_count)/total_count
            score = accuracy*TPR+(1-accuracy)*FNR
            true_rate = TPR
            false_rate = FNR

        elif args.label == "negative":
            correct_count = np.sum(flat_predictions == flat_true_labels)
            total_count = flat_predictions.shape[0]
            TNR = 1.0*correct_count/total_count
            FPR = 1.0*(total_count-correct_count)/total_count
            score = accuracy*TNR+(1-accuracy)*FPR
            true_rate = TNR
            false_rate = FPR

        accuracies_df = accuracies_df.append({
            "dataset": args.input_file,
            "seed_val": seed_val,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "true_rate": true_rate,
            "false_rate": false_rate
        }, ignore_index=True)
        
        iprint(f"Accuracy: {accuracy}")
        # iprint(f"Score: {score}")

    print(accuracies_df)

    
