import torch 
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader 
from transformers import GPT2Tokenizer
import sys 


def dataset_split(dataset_path, output_path): 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    clip_model, preprocess = clip.load("ViT-B/32", device=device) 

    annotation_path = os.path.join(dataset_path, 'annotations/captions_train2014.json') 
    with open(annotation_path, 'r') as f: 
        data = json.load(f)['annotations']
    print("%0d captions loaded from json." %len(data)) 

    all_embeddings = [] 
    all_captions = [] 
    for i in tqdm(range(len(data))):
        d = data[i] 
        img_id = d['image_id'] 
        file_name = os.path.join(dataset_path, f"train2014/COCO_train2014_{int(img_id):012d}.jpg") 
        image = io.imread(file_name) 
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            img_features = clip_model.encode_image(image).cpu() 
        d['clip_embedding'] = i 
        all_embeddings.append(img_features) 
        all_captions.append(d) 
        if i == 20: 
            break 
    
    with open(output_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f) 
    return 0 



class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item] 
        if self.padding == False:
            padding = 0
        else:
            padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float() 
        return tokens, mask

    def __getitem__(self, item: int):
        tokens, mask = self.pad_tokens(item)
        features = self.features[self.caption2embedding[item]]
        if self.normalize_prefix:
            features = features.float()
            features = features / features.norm(2, -1)
        return tokens, mask, features

    def __init__(self, data_path: str,  tokenizer, padding=True, normalize_features=False):
        self.tokenizer = tokenizer
        self.normalize_prefix = normalize_features
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.features = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        self.padding=padding
        
        self.captions_tokens = []
        self.caption2embedding = []
        max_seq_len = 0
        for caption in captions_raw:
            self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
            self.caption2embedding.append(caption["clip_embedding"])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))



if __name__ == '__main__': 
    dataset_path='/Users/feizhengcong/Desktop/COCO'
    output_path = './data/train.pkl'
    # dataset_split(dataset_path, output_path)  
    tokenizer = GPT2Tokenizer.from_pretrained('ckpt/gpt2')
    dataset = ClipCocoDataset('data/train.pkl', tokenizer)
    tokens, mask, features = dataset[0] 
    print(tokens, mask)
    print(features.size())
