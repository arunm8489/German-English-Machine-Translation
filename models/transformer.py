import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """

        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_model_dim
        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        # x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x



# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class EmbeddingLayer(nn.Module):
    """
    created an embedding layers
    Args:
        vocab_size: size of vocabulary
        emb_size: size of embedding
    """
    def __init__(self, vocab_size, emb_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        
        
class SeqtoSeqTransformer(nn.Module):

  
    def __init__(self, 
                 src_vocab_size,
                 target_vocab_size,
                 max_len_src,
                 max_len_target,
                 embedding_size,
                 no_of_heads,
                 no_of_encoders,
                 no_of_decoders,
                 pad_idx,
                 drop_out,
                 no_fwd_expansion,
                 device):
        super(SeqtoSeqTransformer, self).__init__()
        """
        Transformer model according to Attention is all you need paper
        Args:
            src_vocab_size: input size vocabulary
            target_vocab_size: target size vocabulary
            max_len_src: maximum length of source sequence
            max_len_target: maximum length of target sequence
            embedding_size: embedding dimenion of source/target
            no_of_heads: number of heads of multi attention heads
            no_of_encoders: number of encoders
            no_of_decoders: number of decoders
            pad_idx: pad index of source
            drop_out = dropout probability
            no_fwd_expansion: number of dense nuerons in forward expansion in encoder
            device: available device - cpu or cuda
        """
        self.src_word_embedding = EmbeddingLayer(src_vocab_size, embedding_size)
       
        self.src_positional_embedding = PositionalEncoding(max_len_src,embedding_size)
        self.target_word_embedding = EmbeddingLayer(target_vocab_size, embedding_size)
        self.target_positional_embedding = PositionalEncoding(max_len_target,embedding_size)
      

        self.transformer = nn.Transformer(d_model=embedding_size,nhead=no_of_heads,num_encoder_layers=no_of_encoders,
                                          num_decoder_layers=no_of_decoders,dim_feedforward=no_fwd_expansion,dropout=drop_out,
                                          batch_first=True)
        
        self.fc = nn.Linear(embedding_size,target_vocab_size)
        self.drop_out = nn.Dropout(drop_out)  
        self.pad_idx = pad_idx
        self.device = device

    def create_padding_mask(self,src,tgt,pad_idx):
        """
        creates source and target padding mask
        """
        src_padding_mask = (src == pad_idx)
        tgt_padding_mask = (tgt == pad_idx)
        return src_padding_mask.to(self.device),tgt_padding_mask.to(self.device)
    
    def forward(self, src, trg):
        batch_size, src_seq_len = src.shape
        batch_size, target_seq_len = trg.shape 
        
        src_embedding_out = self.src_positional_embedding(self.src_word_embedding(src))
        trg_embedding_out = self.target_positional_embedding(self.target_word_embedding(trg))
       
        # create source and target masks
        src_padding_mask,tgt_padding_mask = self.create_padding_mask(src,trg,self.pad_idx)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)
        target_mask = self.transformer.generate_square_subsequent_mask(target_seq_len).to(self.device)
    
        # transformer
        out = self.transformer(src_embedding_out,trg_embedding_out,
                               src_mask = src_mask,
                               tgt_mask = target_mask,
                               memory_mask = None,
                               src_key_padding_mask = src_padding_mask,
                               tgt_key_padding_mask = tgt_padding_mask,
                              #  memory_key_padding_mask  = src_padding_mask
                               )
        
        out = self.fc(out)
        return out
    
    def encoder(self,src):
        """
        returns encoder output
        """
        src_seq_len = src.shape[1]
        src_embedding_out = self.src_positional_embedding(self.src_word_embedding(src))
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)
        return self.transformer.encoder(src_embedding_out,src_mask)

    def decode(self,trg,encoder_out):
        """
        returns decoder output
        """
        target_seq_len = trg.shape[1]
        target_mask = self.transformer.generate_square_subsequent_mask(target_seq_len).to(self.device)
        trg_embedding_out = self.target_positional_embedding(self.target_word_embedding(trg))
        return self.transformer.decoder(trg_embedding_out,encoder_out,target_mask)
