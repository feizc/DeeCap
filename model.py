from transformers import * 
import math 
import torch 
import torch.nn as nn 
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
        self.threshold = 1.0 

        self.img_ff = nn.Linear(2048, config.n_embd) 
        self.img_inverse_ff = nn.Linear(config.n_embd, 2048) 

        self.base_models = nn.ModuleList(base_models) 
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False)
                                        for _ in range(len(base_models))])
        self.init_params() 

    def init_params(self):
        for name, module in self.named_children(): 
            if 'base_models' not in name:
                module.apply(self._init_weights)
            self._init_weights(self) 
        self.tie_weights() 
    
    def set_threshold(self, threshold):
        self.threshold = threshold 

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

