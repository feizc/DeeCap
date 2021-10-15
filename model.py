from transformers import * 
import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn import CrossEntropyLoss, MSELoss 
from typing import List 


class DeeCapModel(GPT2PreTrainedModel): 
    def __init__(self, config): 
        super(DeeCapModel, self).__init__(config) 
        self.transformer = GPT2Model(config) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.img_ff = nn.Linear(2048, config.n_embd) 
        self.img_inverse_ff = nn.Linear(config.n_embd, 2048) 

        self.init_weights() 
        self.tie_weights() 

    def tie_weights(self): 
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte) 

    def forward(self, input_embs, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None): 
        transformer_outputs = self.transformer(inputs_embeds=input_embs,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask) 
        hidden_states = transformer_outputs[0] 
        lm_logits = self.lm_head(hidden_states) 
        outputs = (lm_logits,) + transformer_outputs[1:] 
        txt_loss_fct = CrossEntropyLoss(ignore_index=-100)
        if labels is not None: 
            loss = txt_loss_fct(lm_logits ,labels) 
            outputs = (loss,) + outputs

        return outputs 



class ModelSequence(GPT2PreTrainedModel):
    def __init__(self, config, base_models: List[GPT2Model]=None,
                 confidence_margin=0.5, margin_loss_weight=1.0):
        super(ModelSequence, self).__init__(config) 
        
        self.vocab_size = config.vocab_size 
        self.threshold = 0.01 

        self.img_ff = nn.Linear(2048, config.n_embd) 
        self.img_inverse_ff = nn.Linear(config.n_embd, 2048) 

        self.base_models = nn.ModuleList(base_models) 
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False)
                                        for _ in range(len(base_models))]) 
        self.margin_loss_weight = margin_loss_weight 
        self.margin_loss = nn.MarginRankingLoss(margin=confidence_margin) 

        self.init_params() 

    def init_params(self):
        for name, module in self.named_children(): 
            if 'base_models' not in name:
                module.apply(self._init_weights)
            self._init_weights(self) 
        self.tie_weights() 
    
    def set_threshold(self, threshold):
        self.threshold = threshold 

    def set_margin_loss_weight(self, margin_loss_weight):
        self.margin_loss_weight = margin_loss_weight 

    def pair_loss(self, difficult_labels, confidence, dif1=0, dif2=1): 
        easy_idx = (difficulty_labels == dif1) 
        hard_idx = (difficult_labels == dif2) 
        easy_conf = confidence[easy_idx] 
        hard_conf = confidence[hard_idx] 

        if len(easy_conf) == 0 or len(hard_conf) == 0:
            return 0.0 
        uniform = torch.ones_like(hard_conf) / len(hard_conf) 
        sampled_hard_idx = torch.multinomial(uniform, num_samples=len(easy_conf), replacement=True) 

        rank_input1 = easy_conf
        rank_input2 = hard_conf[sampled_hard_idx]
        diff_label1 = 1.0 / (1.0 + difficulty_labels[easy_idx])  # 1.0
        diff_label2 = 1.0 / (1.0 + difficulty_labels[hard_idx][sampled_hard_idx])  # 0.5

        geq = torch.where(diff_label1 >= diff_label2, torch.ones_like(diff_label1), torch.zeros_like(diff_label1))
        less = torch.where(diff_label1 < diff_label2, -1 * torch.ones_like(diff_label2), torch.zeros_like(diff_label2))
        target = geq + less
        confidence_loss = self.margin_loss(rank_input1, rank_input2, target)
        return confidence_loss

    def forward(
        self,
        input_embs,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None, 
        head_mask=None,
        labels=None
    ): 
        input_shape=input_embs.size()

        loss = None  
        sequence_outputs = [] 

        for model in self.base_models: 
            outputs = model(
                inputs_embeds=input_embs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask
            )
            sequence_outputs.append(outputs) 
        
        sequence_logits = [] 

        for lm_head, outputs in zip(self.lm_heads, sequence_outputs): 
            outputs = outputs[0] 
            logits = lm_head(outputs) 
            sequence_logits.append(logits) 
        
        if labels is not None: 
            for i, logits in enumerate(sequence_logits): 
                token_logits = logits[-1, :]
                prob = F.softmax(token_logits, dim=-1) 
                confidence, _ = torch.max(prob, dim=-1) 
                if confidence > self.threshold: 
                    return logits 

