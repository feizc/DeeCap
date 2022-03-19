import torch 
from models.transformer import Encoder, Decoder

if __name__ == '__main__': 
    input_f = torch.randn((5,16,512)) 
    input_l = torch.ones((5,20)).long()
    v_encoder = Encoder()
    l_decoder = Decoder() 
    enc_o = v_encoder(input_f)
    print(enc_o[0].size())
    dec_o = l_decoder(input_l, enc_o[0], None)
    print(dec_o[0].size())
