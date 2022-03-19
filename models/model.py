import torch 
from torch import nn 
from torch import Tensor
from .transformer import Encoder, Decoder, ScaledDotProductAttentionMemory, MeshedDecoder
from .containers import Module
from .beam_search import * 


# original transformer image captioning model 
class TICModel(Module): 
    def __init__(self, config):
        super(TICModel, self).__init__() 
        self.model_d = config.n_embd 
        self.clip_dim = config.clip_dim 
        self.clip_length = config.clip_length
        self.feature_project = nn.Linear(config.clip_dim, config.clip_length*config.n_embd) 
        self.visual_encoder = Encoder(config.n_layer, config.clip_length, config.n_embd) 
        self.language_decoder = Decoder(config.vocab_size) 

        self.bos_idx = config.bos_token_id 
        self.eos_idx = config.eos_token_id 
        self.vocab_size = config.vocab_size 
        self.max_generation_length = self.language_decoder.max_len 

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights() 
    
    def init_weights(self):
        for p in self.visual_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.language_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    def forward(self, images, seq):
        images = self.feature_project(images).view(-1, self.clip_length, self.clip_dim)
        enc_output, mask_enc = self.visual_encoder(images)
        dec_output = self.language_decoder(seq, enc_output, mask_enc)
        return dec_output 

    def step(self, t: int, prev_output: Tensor, visual: Tensor) -> Tensor:
        if t == 0:
            visual = self.feature_project(visual).view(-1, self.clip_length, self.clip_dim)
            self.enc_output, self.mask_enc = self.visual_encoder(visual)
            input = visual.data.new_full((visual.shape[0], 1), self.bos_idx, dtype=torch.long)
        else:
            input = prev_output
        logits = self.language_decoder(input, self.enc_output, self.mask_enc)
        return logits 
    
    def beam_search(self, visual, beam_size: int, out_size=1,
                    return_logits=False, **kwargs):
        bs = BeamSearch(self, self.max_generation_length, self.eos_idx, beam_size)
        return bs.apply(visual, out_size, return_logits, **kwargs)



