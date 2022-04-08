from torch.utils.data import Dataset, DataLoader
from utils.utils import clean_ger_text,clean_eng_text
import torch


class LoadTextDataset(Dataset):
    def __init__(self, df, vocab_input,vocab_target, input_seq_length=None, target_seq_length=None, padding=False):
        """
        Loads data from dataframe
        Args:
            df : dataframe
            vocab_input: dictionary of iput text vocabulary
            vocab_target: dictionary of target text vocabulary
            input_seq_length: length of input sequence without SOS and EOS token
            target_seq_length: length of target sequence without SOS and EOS token
            padding: Do we need padding or not
        """    
        self.df = df
        self.padding = padding
        self.input_text_values = df['Ger'].values
        self.target_text_values = df['English'].values
        self.vocab_input = vocab_input
        self.vocab_target = vocab_target
        self.input_seq_length = input_seq_length
        self.target_seq_length = target_seq_length

    def __len__(self):
        """
        returns length of dataframe for dataloader
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Args:
            idx: index
        """
      
        input_text,target_text = self.input_text_values[idx],self.target_text_values[idx]
        
        #preprocessing text
        input_text,target_text = clean_ger_text(input_text),clean_eng_text(target_text)

        text_numeric_input = self.vocab_input.generate_numeric_tokens(input_text)
        text_numeric_target = self.vocab_target.generate_numeric_tokens(target_text)
        text_numeric_input = self.vocab_input.add_eos_sos(text_numeric_input)
        text_numeric_target = self.vocab_target.add_eos_sos(text_numeric_target)

        if self.padding:
          text_numeric_input = self.vocab_input.pad_sequence(text_numeric_input,self.input_seq_length+2)
          text_numeric_target = self.vocab_target.pad_sequence(text_numeric_target,self.target_seq_length+2)
        


        input_seq,target_seq = torch.Tensor(text_numeric_input).long(),torch.Tensor(text_numeric_target).long()

        return input_seq,target_seq