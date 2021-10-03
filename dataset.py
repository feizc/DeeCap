import torch  
import json 
import os 
import h5py 
from torch.utils.data import Dataset 
from transformers import GPT2Tokenizer 



SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[SEP]', '[IMG]', '[TXT]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[IMG]', '[TXT]', '[SEP]'], 'pad_token':'[PAD]'}


class SE2Dataset(Dataset): 

    def __init__(self, data_path, tokenizer):
        self.img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'), 'r')
        train_data_path = os.path.join(data_path, 'annotations')
        with open(os.path.join(train_data_path, 'captions_train2014.json')) as f:
            self.train_data = json.load(f)['annotations'] 
        self.tokenizer = tokenizer 

    def __getitem__(self, i):
        cap_dict = self.train_data[i]
        img_id = str(cap_dict['image_id']) + '_features' 
        img = self.img_features[img_id]
        txt = cap_dict['caption']
        instance = build_input_from_segments(img, txt, self.tokenizer)
        
        input_ids = torch.Tensor(instance['input_ids']).long()
        token_type_ids = torch.Tensor(instance['token_type_ids']).long() 
        lm_labels = torch.Tensor(instance['lm_labels']).long()
        img_features = torch.FloatTensor(img)

        return img_features, input_ids, token_type_ids, lm_labels 

def build_input_from_segments(img_features, caption, tokenizer): 
    bos, eos, sep, img, txt = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    instance = {} 
    caption_id = tokenizer.tokenize(caption)
    caption_id = tokenizer.convert_tokens_to_ids(caption_id)
    input_ids = [sep, bos] + caption_id
    instance['input_ids'] = input_ids 
    token_type_ids = [img] * len(img_features) + [txt]*len(input_ids)
    instance['token_type_ids'] = token_type_ids 
    lm_labels = [-1]*len(img_features) + [bos] + caption_id + [eos] 
    instance['lm_labels'] = lm_labels 
    return instance 


if __name__ == "__main__": 
    data_path = 'data' 
    gpt_model_path = 'model/origin_gpt' 
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path, do_lower_case=True) 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    dataset = SE2Dataset(data_path, tokenizer) 
    img_feature, txt, _, _ = dataset[0] 
    print(txt)
    print(img_feature)