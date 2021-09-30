import torch  
import json 
import os 
import h5py 
from torch.utils.data import Dataset 
from transformers import GPT2Tokenizer 


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
        img = torch.FloatTensor(self.img_features[img_id])
        txt = cap_dict['caption']
        caption = self.tokenizer.tokenize(txt)
        caption_id = self.tokenizer.convert_tokens_to_ids(caption) 
        input_id = torch.Tensor(caption_id).long()

        return img, input_id 


if __name__ == "__main__": 
    data_path = 'data' 
    tokenizer = GPT2Tokenizer.from_pretrained('model/origin_gpt', do_lower_case=True)
    dataset = SE2Dataset(data_path, tokenizer) 
    img_feature, txt = dataset[0] 
    print(txt)
    print(img_feature.size())