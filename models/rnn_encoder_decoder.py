
import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class EncoderRNN(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,no_layers=1,bidirectional=False,drop_out=0.2):
        super(EncoderRNN, self).__init__()
        """
        Args:
            vocab_size: input vocab size
            embed_dim: embedding dimension for input
            hidden_dim: hidden dimension for RNN
            no_layers: number of layers required in RNN
            bidirectional: if True will use bidirectional RNN in encoder
            drop_out = drop out probability
        """
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=no_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=drop_out)
        self.drop_out = nn.Dropout(drop_out)
        self.bidirectional = bidirectional
        self.no_layers = no_layers
        if self.bidirectional:
            self.hidden_fc = nn.Linear(hidden_dim*2,hidden_dim)
            

    def forward(self,x):
        """
        Args:
            x : input sequence of dimension (batch_size x sequence length)
        Returns:
            hidden_state: hidden_state
        """
        batch_size = x.shape[0]  #[4,22] #[batch_size,seq_len]
        embedding_out = self.drop_out(self.embedding(x))  #[4,22,64] #[batch_size,embedding_dim]
        rnn_out, hidden_state = self.rnn(embedding_out)  #[4,22,256] #[1,4,256] 
        # we will pass hidden sate and cell state of last time step to the dnecoder
        if self.bidirectional:
            hidden_state_backward, hidden_state_forward = hidden_state[-1,:,:],hidden_state[-2,:,:]
            hidden_state = torch.concat((hidden_state_forward, hidden_state_backward),axis=1).unsqueeze(0)
            hidden_state = self.hidden_fc(hidden_state)
        return hidden_state 


class DecoderRNN(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,output_size,no_layers=1,bidirectional=False,drop_out=0.2):
        super(DecoderRNN, self).__init__()
        """
        Args:
            vocab_size: input vocab size
            embed_dim: embedding dimension for input
            hidden_dim: hidden dimension for RNN
            output_size: size of output.It is same as vocabulary dimension
            no_layers: number of layers required in RNN
            drop_out = drop out probability
        """

        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim,num_layers=no_layers,batch_first=True,
        bidirectional= bidirectional,dropout=drop_out)
        self.drop_out = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_dim,output_size)

    def forward(self, x, hidden_state):
        """
        Args:
          x : input sequence of size (batch_size x sequence length)
          hidden_state: hiddden state
        Returns:
          predictions: logits predicts of each target word
          hidden_state: hiddden state
        """
        
        # adding an extra dimesnion 
        x = x.unsqueeze(1)  #[4,1]   #[batch_size, single_token]
        
        embedding_out = self.drop_out(self.embedding(x))  #[4,1,64]
        
        rnn_out, hidden_state = self.rnn(embedding_out,hidden_state) #[4,1,256] #[1,4,256] #[1,4,256]
        out = hidden_state[-1:,:,:]  #taking the out from last layer if bidirectional
        
        out = self.fc(out)  #[1, 4, no_of_target_words]

        predictions = out.squeeze(0) #[ 4, no_of_target_words]

        return predictions, hidden_state


class SeqtoSeqRNN(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_len, device):
        super(SeqtoSeqRNN, self).__init__()
        """
        Args:
          encoder: encoder object
          decoder: decoder object
          target_vocab_length: length of target vocabulary
          device: either cpu or cuda

        """

        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_len = target_vocab_len
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        """
        Args:
          source: input batch
          target: target batch
          teacher_force_ratio: ratio which helps in determine what to pass in next iteration, predicted label or actual label
        Returns:
          output: output of the model

        """
        batch_size = source.shape[0]

        target_len = target.shape[1]
   
        hidden_state = self.encoder(source)

        x = target[:,0]
        
        output = torch.zeros(target_len,batch_size,self.target_vocab_len).to(self.device)
        for t in range(1,target_len):
            out, hidden_state = self.decoder(x, hidden_state)   #[4,target_vocab_len]
            
            output[t] = out
            predicted_word = out.argmax(1)
            
            if random.random() < teacher_force_ratio:   #50% of time we take predicted one and in other time actual value
                x = target[:,t]
            else:
                x = predicted_word
            
        return output

