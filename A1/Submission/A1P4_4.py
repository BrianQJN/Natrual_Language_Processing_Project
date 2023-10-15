import random

def tokenize_and_preprocess_text(textlist, w2i, window):
    X, T, Y = [], [], []

    # 1. Tokenize the input
    tokenized_text = [w2i[word] if word in w2i else w2i["<oov>"] for word in textlist]
    
    # 2. Loop through each token
    for idx, token in enumerate(tokenized_text):
        # Positive samples
        start = max(0, idx - window + 1)
        end = min(len(tokenized_text), idx + window)
        
        for j in range(start, end):
            if j != idx:  # Make sure not to include the word itself
                X.append(token)
                T.append(tokenized_text[j])
                Y.append(1)  # Positive sample
        
        # Negative samples
        for _ in range(end - start - 1):  # Minus one to exclude the word itself
            random_token = random.choice(tokenized_text)
            while random_token == token:  # Ensure random word is different from the current word
                random_token = random.choice(tokenized_text)
            X.append(token)
            T.append(random_token)
            Y.append(-1)  # Negative sample

    print(f"Total number of examples (both positive and negative): {len(X)}")
    return X, T, Y

# Define window size
window_size = 5

# 2. Generate Training Samples
X, T, Y = tokenize_and_preprocess_text(filtered_lemmas, w2i, window_size)

# 3. Examine the results
print("First 10 target tokens:", X[:10])
print("First 10 context tokens:", T[:10])
print("First 10 labels (1 for positive, -1 for negative):", Y[:10])

# Check total number of examples
print(f"Total number of examples: {len(X)}")

