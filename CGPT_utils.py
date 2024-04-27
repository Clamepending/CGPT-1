import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        
        frequency_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position*frequency_term)
        pe[:, 1::2] = torch.cos(position*frequency_term)
        
        pe = pe.unsqueeze(0) # add batch dimention
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        pe = self.pe.detach()  # Detach the positional encoding tensor
        x = x + pe[:, :x.shape[1], :]
        return self.dropout(x)
        
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha*(x - mean)/(std + self.eps) + self.beta
    


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff, d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model%num_heads == 0, "num heads does not divide d_model"
        self.d_k = d_model // num_heads
        
        self.Wq = nn.Linear(d_model, d_model) # vec to query
        self.Wk = nn.Linear(d_model, d_model) # vec to key
        self.Wv = nn.Linear(d_model, d_model) # vec to value
        
        self.dropout = nn.Dropout(dropout)
        
        self.Wo = nn.Linear(d_model, d_model)
        
        
    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout, mask = None):
        
        # attention matrix
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(query.shape[-1])
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        scores = torch.softmax(scores, dim = -1)
        
        
        if dropout is not None:
            scores = dropout(scores)
        
        return torch.matmul(scores, value), scores # return the output of the head as well as attention matrix for visualization
        

    def forward(self, q, k, v, mask):
        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)
        
        
        # divide the input vectors into different heads
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.d_k).transpose(1,2)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.d_k).transpose(1,2)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiheadAttentionBlock.attention(Q, K, V, self.dropout, mask)
        
        # print(f"shapes of attentions: {x.shape[0]} {x.shape[1]} {x.shape[2]} {x.shape[3]}")
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_heads*self.d_k)
        
        return self.Wo(x)
        
        
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.ModuleList):
    def __init__(self, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x))
        
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
        
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
        
        

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        
        super().__init__()
        
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x
        
        
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        
        self.Layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.Layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
        
        return self.norm(x)
    
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim = -1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, encoder: Encoder, tgt_embed: InputEmbedding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    
    def decode(self, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.encoder(tgt, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model:int = 512, N:int = 6, h:int = 8, dropout:float = 0.1, dff:int = 2048):
    # embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    
    print("VOCAB SIZES", src_vocab_size, tgt_vocab_size)
    
    # positional encodings
    
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_fwd_block = FeedForwardBlock(d_model, dff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_fwd_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_fwd_block = FeedForwardBlock(d_model, dff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_fwd_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer
        
def build_decoder_only_transformer(vocab_size: int, seq_len: int, d_model:int = 512, N:int = 6, h:int = 8, dropout:float = 0.1, dff:int = 2048):
    # embedding layers
    embed = InputEmbedding(d_model, vocab_size)
    
    # positional encodings
    pos = PositionalEncoding(d_model, seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_fwd_block = FeedForwardBlock(d_model, dff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_fwd_block, dropout)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, vocab_size)
    
    transformer = DecoderOnlyTransformer(encoder, embed, pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer    
    

    
    
    
        
    
        
        
        

        
    
    