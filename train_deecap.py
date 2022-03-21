import torch 
import os 
import argparse 
import random 
import numpy as np 
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss


from models import DeeCapModel, TransformerConfig
from dataset import ClipCocoDataset 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 



use_device = torch.cuda.is_available()
device = torch.device('cuda:0' if use_device else 'cpu') 
torch.backends.cudnn.benchmark = True

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def LayerWeightLoss(model, outputs, labels, args):
    total_loss = None 
    total_weights = 0 
    loss_fct = CrossEntropyLoss() 

    # weight cross-entropy for different language layers 
    for idx, logits_item in enumerate(outputs): 
        loss = loss_fct(logits_item.view(-1, model.config.vocab_size), labels.view(-1)) 
        if total_loss is None: 
            total_loss = loss 
        else:
            total_loss += loss * (idx + 1) 
        total_weights += idx + 1 
    
    # cos similarity loss for hidden representation prediction 
    cos_loss_total = 0 
    if args.predictor_training: 
        hidden_states_list = model.hidden_states_list 
        hidden_states_list = torch.stack(hidden_states_list, dim=2) # (bsz, seq_len, num_layer, model_d)
        hidden_states_proj_list = model.hidden_states_proj_list # (bsz, seq_len, num_layer, model_d)
        
        cos_loss_fct = CosineEmbeddingLoss()
        
        for i in range(len(hidden_states_proj_list) - 1): 
            hidden_states = hidden_states_list[:, :, i+1:, :].reshape(-1, model.model_d) 
            hidden_states_proj = hidden_states_proj_list[i][:, :, i+1:, :].reshape(-1, model.model_d) 
            target = torch.ones(hidden_states.shape[0], device=hidden_states.device) 
            cos_loss_total += cos_loss_fct(hidden_states, hidden_states_proj, target) 
        
        cos_loss_total /= len(hidden_states_proj_list) - 1 

    return total_loss / total_weights + cos_loss_total 



def train(model, train_dataloader, args, optimizer, scheduler, epoch):
    model.train()
    running_loss = .0 
    print('Num Training Epochs = ', epoch)
    progress = tqdm(total=len(train_dataloader), desc='DeeCapModel') 
    for idx, (tokens, _, img_features) in enumerate(train_dataloader):  
        model.zero_grad() 
        tokens, img_features = tokens.to(device), img_features.to(device, dtype=torch.float32) 
        outputs = model(img_features, tokens) 
        loss = LayerWeightLoss(model, outputs, tokens, args)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        progress.set_postfix({"loss": running_loss / (idx + 1)})
        progress.update()
        
    progress.close()
    return running_loss / len(train_dataloader)



def evaluate_loss(model, test_dataloader, args): 
    model.eval() 
    running_loss = .0 
    progress = tqdm(total=len(test_dataloader), desc='DeeCapModel') 
    with torch.no_grad():
        for idx, (tokens, _, img_features) in enumerate(test_dataloader):  
            tokens, img_features = tokens.to(device), img_features.to(device, dtype=torch.float32) 
            outputs = model(img_features, tokens) 
            loss = LayerWeightLoss(model, outputs, tokens, args)
        
            running_loss += loss.item()
            progress.set_postfix({"loss": running_loss / (idx + 1)})
            progress.update() 
    val_loss = running_loss / len(test_dataloader) 
    return val_loss 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train_data_path', default='./data/train.pkl')
    parser.add_argument('--test_data_path', default='./data/test.pkl')
    parser.add_argument('--tokenizer_path', default='./ckpt/gpt2') 
    parser.add_argument('--batch_size', default=5) 
    parser.add_argument('--lr', default=1e-4) 
    parser.add_argument('--epochs', default=8) 
    parser.add_argument('--warmup_steps', default=5000) 
    parser.add_argument('--out_dir', default='./ckpt') 
    parser.add_argument('--model_type', default='deecap') 
    parser.add_argument('--predictor_training', default=True) 
    args = parser.parse_args() 

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path) 
    train_dataset = ClipCocoDataset(args.train_data_path, tokenizer) 
    test_dataset = ClipCocoDataset(args.test_data_path, tokenizer) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) 
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False) 

    config = TransformerConfig()
    model = DeeCapModel(config).to(device) 

    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    ) 

    for epoch in range(args.epochs): 
        train(model, train_dataloader, args, optimizer, scheduler, epoch) 
        val_loss = evaluate_loss(model, test_dataloader, args)

        torch.save(
            model.state_dict(),
            os.path.join(args.out_dir, '%s_last.pth' %args.model_type)
        )
        break 



if __name__ == "__main__": 
    main()