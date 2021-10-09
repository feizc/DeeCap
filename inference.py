from transformers import * 
import torch  
import torch.nn.functional as F 
import numpy as np 

from model import SE2Model 
from dataset import build_input_from_segments 

SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[SEP]', '[IMG]', '[TXT]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[IMG]', '[TXT]', '[SEP]'], 'pad_token':'[PAD]'}






if __name__ == '__main__': 
    ckpt_model_path = 'model/ckpt' 
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_model_path, do_lower_case=True) 
    model_config = GPT2Config.from_pretrained(ckpt_model_path)
    model = SE2Model(model_config) 
    
    ckpt = torch.load('model/ckpt/epoch_0', map_location='cpu') 
    model.load_state_dict(ckpt['model']) 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()



