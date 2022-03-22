from transformers import * 
import torch  
import torch.nn.functional as F 
import numpy as np 
import copy 

from model import DeeCapModel 
from dataset import build_input_from_segments, DeeCapDataset 

SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[SEP]', '[IMG]', '[TXT]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[IMG]', '[TXT]', '[SEP]'], 'pad_token':'[PAD]'}


def beam_search(img_features, model, tokenizer, max_length=25, beam_size=5): 
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    current_output = '' 
    hyplist = [([], 0., current_output)] 
    comp_hyplist = [] 
    img_emb = model.img_ff(img_features)
    for i in range(max_length): 
        new_hyplist = [] 
        argmin = 0
        for out, lp, st in hyplist: 
            instance = build_input_from_segments(img_features, st, tokenizer, label_flag=False) 
            input_ids = torch.tensor(instance['input_ids']).long() 
            token_type_ids = torch.tensor(instance['token_type_ids']).long() 
            input_emb = model.transformer.wte(input_ids)
            input_embs = torch.cat([img_emb, input_emb], dim=0) 
            # print(input_embs.size(), toekn_type_ids.size()) 

            logits = model(input_embs=input_embs, token_type_ids=token_type_ids)[0]
            logp = F.log_softmax(logits, dim=-1)[-1, :] 
            lp_vec = logp.cpu().data.numpy() + lp 
            
            for o in np.argsort(lp_vec)[::-1]:
                if o == tokenizer.unk_token_id or o == tokenizer.eos_token_id: 
                    continue 
                new_lp = lp_vec[o] 
                if len(new_hyplist) == beam_size: 
                    if new_hyplist[argmin][1] < new_lp: 
                        new_st = copy.deepcopy(st)
                        new_st += ' '
                        new_st += tokenizer.convert_ids_to_tokens([o])[0]
                        new_hyplist[argmin] = (out+[o], new_lp, new_st) 
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1])[0]
                    else:
                        break 
                else:
                    new_st = copy.deepcopy(st)
                    new_st += ' '
                    new_st += tokenizer.convert_ids_to_tokens([o])[0]
                    new_hyplist.append((out+[o], new_lp, new_st))
                    if len(new_hyplist) == beam_size: 
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1])[0]
        hyplist = new_hyplist 
    maxhyps = sorted(hyplist, key=lambda h: -h[1])[:1]
    print(maxhyps)



def generate_caption(model, tokenizer, data): 
    model.eval()
    with torch.no_grad():
        for instance in data: 
            img_features = instance[0]
            hypstr = beam_search(img_features, model, tokenizer)
            break 


if __name__ == '__main__': 
    ckpt_model_path = 'model/ckpt' 
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_model_path, do_lower_case=True) 
    model_config = GPT2Config.from_pretrained(ckpt_model_path)
    model = DeeCapModel(model_config) 
    
    ckpt = torch.load('model/ckpt/epoch_1', map_location='cpu') 
    model.load_state_dict(ckpt['model']) 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    test_dataset_path = 'data'
    test_dataset = DeeCapDataset(test_dataset_path, tokenizer)  

    generate_caption(model, tokenizer, test_dataset) 




