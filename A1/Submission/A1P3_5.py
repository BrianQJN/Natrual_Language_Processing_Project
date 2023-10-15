import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from . import A1P3_4


def train_word2vec(textlist, window=5, embedding_size=2):
    
    # 1. Create the training data
    X, Y = tokenize_and_preprocess_text(textlist, v2i, window)
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    
    # 2. Split the training data: 80% for training and 20% for validation
    dataset = TensorDataset(X, Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 3. Instantiate the network
    network = A1P3_4.Word2vecModel(vocab_size=len(v2i), embedding_size=embedding_size)
    
    # 4. Set up the optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)

    # Training setup
    epochs = 50
    train_losses = []
    val_losses = []

    # 5. Training loop
    for epoch in range(epochs):
        network.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits, _ = network(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        
        # Validation loss
        network.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                logits, _ = network(x_batch)
                loss = F.cross_entropy(logits, y_batch)
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(val_loader))
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plotting the training and validation curves
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return network

def tokenize_and_preprocess_text(textlist, v2i, window):

    X, Y = [], []  # List of training/test samples
    
    # Determine the actual context size from the window.
    context_size = (window - 1) // 2 

    # Iterate over the words in the textlist.
    for i, word in enumerate(textlist):
        
        # Define the start and end index for the context words.
        start_index = max(0, i - context_size)
        end_index = min(len(textlist), i + context_size + 1)
        
        # Iterate over the context words.
        for j in range(start_index, end_index):
            
            # Skip if the context word is the same as the target word.
            if i != j:
                X.append(v2i[word])
                Y.append(v2i[textlist[j]])

    return X, Y

train_word2vec(lemmas, window=5, embedding_size=2)