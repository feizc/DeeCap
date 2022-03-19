import torch 
from torch import nn 
from .containers import Module, ModuleList
from .transformer import DecoderLayer, sinusoid_encoding_table, Encoder
from .utils import one_hot_to_index 



def entropy(x):
    prob = torch.nn.functional.softmax(x, dim=1)
    return -torch.sum(prob * torch.log(prob), dim=1)



class DeeCapPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the last hidden state 
        last_token_tensor = hidden_states[:, -1]
        pooled_output = self.dense(last_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output




class InternalClassifier(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.n_layer, config.n_embd, config.n_embd).
                                   normal_(mean=0.0, std=config.initializer_range))
        self.bias = nn.Parameter(torch.zeros(config.n_layer, config.n_embd))
        self.act = nn.Tanh()

    def forward(self, hidden_representation):
        confidence_pred = self.act(hidden_representation.matmul(self.weight).permute(1, 0, 2) + self.bias)
        return confidence_pred




class DeeCapDecoder(Module):
    def __init__(self, vocab_size=50257, max_len=40, N_dec=3, padding_idx=0, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, self_att_module=None, enc_att_module=None, self_att_module_kwargs=None,
                 enc_att_module_kwargs=None):
        super(DeeCapDecoder, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                          enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                          enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', None)
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        input = input[:, :self.max_len]
        b_s, seq_len = input.shape[:2]

        if input.dtype in [torch.long, torch.int]:
            input_index = input
        else:
            input_index = one_hot_to_index(input)

        mask_queries = (input_index != self.padding_idx).unsqueeze(-1).type(input.dtype)  
        # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input.device),
                                         diagonal=1)
        # print(mask_self_attention)  (seq_len, seq_len)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input_index == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            if self.running_mask_self_attention is None:
                self.running_mask_self_attention = mask_self_attention
            else:
                self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention],
                                                             -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        if input.dtype in [torch.long, torch.int]:
            out = self.word_emb(input)
        else:
            out = input @ self.word_emb.weight

        out = out + self.pos_emb(seq)
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return out 
    
    # only forward propagation with transformer block 
    def adaptive_forward(self, hidden_states, current_layer, encoder_output, mask_queries, mask_self_attention, mask_encoder): 
        layer_outputs = self.layers[current_layer](hidden_states, encoder_output, mask_queries, mask_self_attention, mask_encoder)
        return layer_outputs 
    
    def word_forward(self, input):
        # input (b_s, seq_len)
        input = input[:, :self.max_len]
        b_s, seq_len = input.shape[:2]

        if input.dtype in [torch.long, torch.int]:
            input_index = input
        else:
            input_index = one_hot_to_index(input)

        mask_queries = (input_index != self.padding_idx).unsqueeze(-1).type(input.dtype)  
        # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input.device),
                                         diagonal=1)
        # print(mask_self_attention)  (seq_len, seq_len)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input_index == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            if self.running_mask_self_attention is None:
                self.running_mask_self_attention = mask_self_attention
            else:
                self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention],
                                                             -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        if input.dtype in [torch.long, torch.int]:
            out = self.word_emb(input)
        else:
            out = input @ self.word_emb.weight

        out = out + self.pos_emb(seq) 
        return out, mask_queries, mask_self_attention 



class DeeCapModel(Module): 
    def __init__(self, config):
        super(DeeCapModel, self).__init__() 
        self.config = config 
        self.model_d = config.n_embd 
        self.clip_dim = config.clip_dim 
        self.clip_length = config.clip_length
        self.feature_project = nn.Linear(config.clip_dim, config.clip_length*config.n_embd) 
        self.visual_encoder = Encoder(config.n_layer, config.clip_length, config.n_embd) 
        self.language_decoder = DeeCapDecoder(config.vocab_size, N_dec=config.n_layer) 
        self.internal_classifier = InternalClassifier(config) 
        self.poolers = nn.ModuleList([DeeCapPooler(config) for _ in range(config.n_layer)])

        self.bos_idx = config.bos_token_id 
        self.eos_idx = config.eos_token_id 
        self.vocab_size = config.vocab_size 
        self.max_generation_length = self.language_decoder.max_len 
        self.freezed_lower_layer = 3 

        self.confidence_tokens = [] 
        self.confidence_tokens_proj = [] 

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

        self.confidence_tokens.clear() 
        self.confidence_tokens_proj.clear() 

        # dec_output = self.language_decoder(seq, enc_output, mask_enc)

        hidden_states, mask_queries, mask_self_attention = self.language_decoder.word_forward(seq) 
        # (bsz, seq_len, model_d)
        res = [] 
        for i in range(self.config.n_layer): 
            
            hidden_states = self.language_decoder.adaptive_forward(hidden_states, i, enc_output, mask_queries, mask_self_attention, mask_enc)
            if i < self.freezed_lower_layer: 
                hidden_states = hidden_states.detach() 
            pooled_output = self.poolers[i](hidden_states)
            confidence_token = pooled_output 

            self.confidence_tokens.append(confidence_token.detach())
            self.confidence_tokens_proj.append(self.internal_classifier(confidence_token))
        
        return res

    





