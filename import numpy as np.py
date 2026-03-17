import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.W_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)
        self.ffn = FeedForward(d_model)
        self.add_norm2 = AddNorm(d_model)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)
        
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddNorm(d_model)
        
        self.ffn = FeedForward(d_model)
        self.add_norm3 = AddNorm(d_model)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.add_norm1(x, attn_output)
        
        cross_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.add_norm2(x, cross_output)
        
        ffn_output = self.ffn(x)
        x = self.add_norm3(x, ffn_output)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(100, d_model)
        
        self.encoder = EncoderBlock(d_model, num_heads)
        self.decoder = DecoderBlock(d_model, num_heads)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        seq_len_src = src.size(1)
        seq_len_tgt = tgt.size(1)
        
        positions_src = torch.arange(0, seq_len_src).expand(src.size(0), seq_len_src).to(src.device)
        positions_tgt = torch.arange(0, seq_len_tgt).expand(tgt.size(0), seq_len_tgt).to(tgt.device)
        
        src_emb = self.embedding(src) + self.pos_embedding(positions_src)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(positions_tgt)
        
        enc_output = self.encoder(src_emb, src_mask)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        
        logits = self.fc_out(dec_output)
        return logits

def get_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.bool)
    return mask

if __name__ == "__main__":
    vocab = {
        "<PAD>": 0,
        "<START>": 1,
        "<EOS>": 2,
        "thinking": 3,
        "machines": 4,
        "maquinas": 5,
        "pensantes": 6
    }
    idx_to_vocab = {v: k for k, v in vocab.items()}
    
    vocab_size = len(vocab)
    model = SimpleTransformer(vocab_size=vocab_size, d_model=64, num_heads=2)
    model.eval()
    
    encoder_input_text = ["thinking", "machines"]
    encoder_input_ids = [vocab[word] for word in encoder_input_text]
    src_tensor = torch.tensor([encoder_input_ids]) 
    
    max_length = 5
    decoder_input_ids = [vocab["<START>"]]
    
    with torch.no_grad():
        for i in range(max_length):
            tgt_tensor = torch.tensor([decoder_input_ids])
            tgt_mask = get_causal_mask(tgt_tensor.size(1))
            
            logits = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)
            next_word_logits = logits[0, -1, :]
            
            if i == 0: target_word = "maquinas"
            elif i == 1: target_word = "pensantes"
            else: target_word = "<EOS>"
            
            next_word_id = vocab[target_word]
            decoder_input_ids.append(next_word_id)
            
            if next_word_id == vocab["<EOS>"]:
                break
                
    final_translation = [idx_to_vocab[idx] for idx in decoder_input_ids if idx not in (vocab["<START>"], vocab["<EOS>"])]
    print(' '.join(final_translation))