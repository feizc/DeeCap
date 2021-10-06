from transformers import * 
import os 
import torch 
import json 
import numpy as np 
from model import SE2Model 
from dataset import SE2Dataset 
from torch.utils.data import DataLoader 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[SEP]', '[IMG]', '[TXT]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[IMG]', '[TXT]', '[SEP]'], 'pad_token':'[PAD]'}

# parameters 
train_dataset_path = 'data' 
use_cuda = torch.cuda.is_available() 
device = torch.device('cuda' if use_cuda else 'cpu') 
gpt_model_path = 'model/origin_gpt' 
ckpt_usage = False 

lr = 6e-3
epochs = 10
gradient_accumulation_steps = 1 
print_freq = 1 


def main(): 
    if ckpt_usage == True: 
        ckpt_path = '' 
    else: 
        tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path, do_lower_case=True) 
        model = SE2Model.from_pretrained(gpt_model_path) 
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
        model.resize_token_embeddings(len(tokenizer)) 
    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr) 

    dataset = SE2Dataset(train_dataset_path, tokenizer) 
    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr) 

    train_dataset = SE2Dataset(train_dataset_path, tokenizer) 
    # train_loader = DataLoader(train_dataset, batch_size=1) 

    for epoch in range(epochs): 
        train(model=model, tokenizer=tokenizer, optimizer=optimizer, dataset=train_dataset, epoch=epoch) 


def train(model, tokenizer, optimizer, dataset, epoch): 
    model.train() 
    iteration = 1 

    for instance in dataset: 
        instance = tuple(input_tensor.to(device) for input_tensor in instance)
        img_features, input_ids, token_type_ids, lm_labels = instance 
        input_emb = model.transformer.wte(input_ids)
        img_emb = model.img_ff(img_features)
        input_embs = torch.cat([img_emb, input_emb], dim=-2) 

        #print(input_embs.size())
        #print(token_type_ids.size())
        
        loss = model(input_embs, token_type_ids=token_type_ids, labels=lm_labels)[0]
        
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        if iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        iteration += 1 
        print(loss)
        break 








if __name__ == '__main__': 
    main()
