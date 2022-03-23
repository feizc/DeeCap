import torch 
import os 
import argparse 
import random 
import numpy as np 
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 


from models.transformer import Encoder, Decoder
from models import entropy, DeeCapModel, TransformerConfig 
from dataset import ClipCocoDataset 
import evaluation


use_device = torch.cuda.is_available()
device = torch.device('cuda:0' if use_device else 'cpu') 
torch.backends.cudnn.benchmark = True

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

SPECIAL_TOKENS = ["<bos>", "<eos>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", }
max_length = 20 


def greedy_decode(img_features, model, tokenizer): 
    special_token_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS) 

    gen_i = [special_token_ids[0]]  

    for i in range(max_length): 
        tokens = torch.tensor(gen_i).long().unsqueeze(0)
        logits = model.step(img_features, tokens) 
        logits = logits[0].cpu().numpy()
        next_word = np.argsort(logits)[-1] 
        if next_word == special_token_ids[1]:
            break
        gen_i.append(next_word) 
    return gen_i 



def predict_captions(model, test_dataloader, tokenizer): 
    import itertools 
    model.eval() 
    gen = {} 
    gts = {} 
    progress = tqdm(total=len(test_dataloader), desc='DeeCapModel') 
    with torch.no_grad():
        for idx, (tokens, _, img_features) in enumerate(test_dataloader):  
            tokens, img_features = tokens.to(device), img_features.to(device, dtype=torch.float32) 
            gen_i = greedy_decode(img_features, model, tokenizer) 
            caps_gen = tokenizer.batch_decode([gen_i])
            caps_gt = tokenizer.batch_decode(tokens) 
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (idx, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (idx, i)] = gts_i
            progress.update() 
            
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_all_scores(gts, gen)
    return scores 



if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='deecap')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_data_path', default='./data/test.pkl')
    parser.add_argument('--tokenizer_path', default='./ckpt/gpt2') 
    args = parser.parse_args() 

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path) 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    test_dataset = ClipCocoDataset(args.test_data_path, tokenizer) 
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False) 

    config = TransformerConfig(vocab_size=len(tokenizer))
    model = DeeCapModel(config).to(device) 
    
    scores = predict_captions(model, test_dataloader, tokenizer) 
    print(scores)


    '''
    input_f = torch.randn((5,512)) 
    input_l = torch.ones((5,20)).long()
    

    configuration = TransformerConfig() 
    model = DeeCapModel(configuration) 
    model.step(input_f, input_l)
    '''
