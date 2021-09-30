from transformers import * 
import math 
import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss, MSELoss 


class SE2Model(GPT2PreTrainedModel): 
    def __init__(self, config): 
        super(SE2Model, self).__init__(config) 
        self.transformer = GPT2Model(config) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.img_ff = nn.Linear(2048, config.n_embd) 
        self.img_inverse_ff = nn.Linear(config.n_embd, 2048) 

        self.init_weights() 
        self.tie_weights() 

    def tie_weights(self): 
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte) 

    def forward(self, input_embs, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None): 
        transformer_outputs = self.transformer(input_embs,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask) 
        hidden_states = transformer_outputs[0] 
        lm_logits = self.lm_head(hidden_states) 
        outputs = (lm_logits,) + transformer_outputs[1:] 
        return outputs 