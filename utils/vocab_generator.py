import en_core_web_sm,de_core_news_sm



class VocabGenerator:
    def __init__(self, corpus, min_frequency=2, tokenizer_lang="english"):
        """
        Args:
           corpus: list of text documents
           min_frequency: mimimum frequency by which a word should be included in corpus
           tokenizer_lang: language of tokenizer
        """
        self.stoi = {'<UNK>':1,'<PAD>':0,'<SOS>':2,'<EOS>':3}
        self.itos = {1:'<UNK>',0:'<PAD>',2:'<SOS>',3:'<EOS>'}
        self.min_freq = min_frequency
      
        #tokenizer
        self.lang = tokenizer_lang
        if tokenizer_lang == "english":
            self.tokenizer = en_core_web_sm.load()
        else:
            self.tokenizer = de_core_news_sm.load()

        self.build_vocabulary(corpus)
    
    def __len__(self):
        """
        Returns length of the vocabulary
        """
        return len(self.itos)
        
    def counter(self,text):
        """
        counts the frequency of words in corpus
        Args:
            text : text for splitting
        """
        
        out = {}
        text = text.split()
        for word in text:
            if word in out:
                out[word] += 1
            else:
                out[word] = 1
        return out
    
    def build_vocabulary(self,corpus):
        """
        function to build vocabulary from corpus.It updates stoi and itos dictionaries
        Args:
            corpus: list of documents
        """
        idx = 4
        frequency_dict = self.counter(corpus)
        for word,freq in frequency_dict.items():
          if freq >= self.min_freq:
              self.stoi[word] = idx
              self.itos[idx] = word
              idx += 1

    def generate_numeric_tokens(self,text):
        """
        generates token from text
        Args:
            text: input text
        """
        
        tokenized_out = [token.text for token in self.tokenizer(text)]
        out = [self.stoi[token] if token in self.stoi.keys() else self.stoi["<UNK>"] for token in tokenized_out]    
        return out

    def add_eos_sos(self,numeric_list):
        """
        add EOS and SOS token to tokens already generated
        Args:
           numeric_list: list of integers generated from token
        """
        numeric_list = [self.stoi['<SOS>']] + numeric_list + [self.stoi['<EOS>']] 
        return numeric_list
        
    def pad_sequence(self,numeric_list,max_seq_length):
        """
        function which do padding and truncation of integer list of tokens
        Args:
            numeric_list: integer list of tokens
            max_seq_length: length to which sequence is to be padded
        """
        if len(numeric_list) < max_seq_length:
            no_zeros = max_seq_length - len(numeric_list)
            numeric_list = numeric_list + [self.stoi['<PAD>'] for i in range(no_zeros)]
        else:
            numeric_list = numeric_list[:max_seq_length]
        
        return numeric_list
        

# if __name__=="__main__":
#     x = "hai iam ok hai ok cool"
