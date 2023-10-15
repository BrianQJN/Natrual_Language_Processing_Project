import torch.nn as nn

class Word2vecModel(nn.Module):
    def __init__(self, vocab_size, embedding_size=2):
        super(Word2vecModel, self).__init__()

        # Embedding layer: Converts token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # Linear layer: Maps from embedding vectors to vocabulary size (used for prediction)
        self.linear = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, x):
        # Lookup the embedding for the input word indices
        e = self.embedding(x)
        
        # Use the embedding to predict the context word
        logits = self.linear(e)
        
        return logits, e