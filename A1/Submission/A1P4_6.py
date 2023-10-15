import torch

class SkipGramNegativeSampling(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        
        # Initialize the embedding layer with `vocab_size` and `embedding_size`
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        
    def forward(self, x, t):
        # x: torch.tensor of shape (batch_size), context word
        # t: torch.tensor of shape (batch_size), target ("output") word.
        
        # Fetch the embeddings for the context and target words
        x_embed = self.embedding(x)
        t_embed = self.embedding(t)
        
        # Compute the dot product for each pair of context and target embeddings
        prediction = torch.sum(x_embed * t_embed, dim=1)
        
        return prediction
