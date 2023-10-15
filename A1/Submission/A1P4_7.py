import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_sgns(textlist, window=5, embedding_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Create Training Data
    X, T, Y = tokenize_and_preprocess_text(textlist, w2i, window)
    X = torch.tensor(X, dtype=torch.long)
    T = torch.tensor(T, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    # 2. Split the training data: 80% for training and 20% for validation
    dataset = TensorDataset(X, T, Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 3. Initialize the model and optimizer
    network = SkipGramNegativeSampling(vocab_size=len(w2i), embedding_size=embedding_size).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=4e-4)  
    
    epochs = 30
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        network.train()
        total_loss = 0
        for x_batch, t_batch, y_batch in train_loader:
            x_batch, t_batch, y_batch = x_batch.to(device), t_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            prediction = network(x_batch, t_batch)
            
            # Custom loss function
            loss = -torch.mean(y_batch * torch.log(torch.sigmoid(prediction) + 1e-5) + 
                              (1 - y_batch) * torch.log(1 - torch.sigmoid(prediction) + 1e-5))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        train_losses.append(total_loss / len(train_loader))
        
        # Validation loss
        network.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, t_batch, y_batch in val_loader:
                prediction = network(x_batch, t_batch)
                loss = -torch.mean(y_batch * torch.log(torch.sigmoid(prediction) + 1e-5) + 
                                  (1 - y_batch) * torch.log(1 - torch.sigmoid(prediction) + 1e-5))
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

# Run the training loop
network = train_sgns(filtered_lemmas)