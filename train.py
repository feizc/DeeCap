from transformers import * 
import os 
import torch 
import json 
import numpy as np 
from torch.utils.data import DataLoader 

from model import SE2Model 
from dataset import SE2Dataset 
from utils import accuracy_compute, AverageMeter 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[SEP]', '[IMG]', '[TXT]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[IMG]', '[TXT]', '[SEP]'], 'pad_token':'[PAD]'}

# parameters 
train_dataset_path = 'data' 
use_cuda = torch.cuda.is_available() 
device = torch.device('cuda' if use_cuda else 'cpu') 
gpt_model_path = 'model/origin_gpt' 
ckpt_model_path = 'model/ckpt'
ckpt_usage = False 

lr = 6e-3
epochs = 1
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

    for epoch in range(1, epochs+1): 
        train(model=model, tokenizer=tokenizer, optimizer=optimizer, dataset=train_dataset, epoch=epoch) 
    
        torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},\
            '%s/epoch_%d'%(ckpt_model_path, epoch))
        model.config.to_json_file(os.path.join(ckpt_model_path, 'config.json'))
        tokenizer.save_vocabulary(ckpt_model_path)


def train(model, tokenizer, optimizer, dataset, epoch): 
    model.train() 
    iteration = 1 
    avg_loss =  AverageMeter() 
    avg_acc = AverageMeter() 

    for instance in dataset: 
        instance = tuple(input_tensor.to(device) for input_tensor in instance)
        img_features, input_ids, token_type_ids, lm_labels = instance 
        input_emb = model.transformer.wte(input_ids)
        img_emb = model.img_ff(img_features)
        input_embs = torch.cat([img_emb, input_emb], dim=-2) 

        #print(tokenizer.convert_ids_to_tokens(input_ids))
        #print(tokenizer.convert_ids_to_tokens(token_type_ids)) 
        #print(tokenizer.convert_ids_to_tokens(lm_labels))
        
        #print(input_embs.size())
        feature_num = img_features.size(0)
        #print(token_type_ids.size())
        

        loss, lm_logits, _ = model(input_embs, token_type_ids=token_type_ids, labels=lm_labels)
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        acc = accuracy_compute(lm_logits[feature_num:,], lm_labels[feature_num:])
        avg_acc.update(acc) 
        avg_loss.update(loss.item())

        if iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        
        if iteration % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(epoch, iteration, len(dataset),loss=avg_loss, acc=avg_acc))

        iteration += 1 


        break 








if __name__ == '__main__': 
    main()
