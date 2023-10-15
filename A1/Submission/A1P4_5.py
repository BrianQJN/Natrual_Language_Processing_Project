from math import sqrt


def subsample_probability(frequency, t=1e-5):
    return min(1, (sqrt(frequency / t) + 1) * (t / frequency))

def tokenize_and_preprocess_text_subsample(textlist, w2i, window):
    """
    Skip-gram negative sampling with subsampling of frequent words.
    """
    X, T, Y = [], [], []

    # Calculate total word count for frequencies
    total_count = len(textlist)
    word_frequencies = {word: count / total_count for word, count in Counter(textlist).items()}
    
    # Tokenized and subsampled text
    subsampled_text = [word for word in textlist if random.random() < subsample_probability(word_frequencies[word])]

    # 2. Loop through each token
    for idx, token in enumerate(subsampled_text):
        # Positive samples
        start = max(0, idx - window + 1)
        end = min(len(subsampled_text), idx + window)
        
        for j in range(start, end):
            if j != idx:  # Make sure not to include the word itself
                X.append(token)
                T.append(subsampled_text[j])
                Y.append(1)  # Positive sample
        
        # Negative samples
        for _ in range(end - start - 1):  # Minus one to exclude the word itself
            random_token = random.choice(subsampled_text)
            while random_token == token:  # Ensure random word is different from the current word
                random_token = random.choice(subsampled_text)
            X.append(token)
            T.append(random_token)
            Y.append(-1)  # Negative sample

    print(f"Total number of examples (both positive and negative): {len(X)}")
    return X, T, Y

# Usage
X, T, Y = tokenize_and_preprocess_text_subsample(filtered_lemmas, w2i, 5)

# Check total number of examples
print(f"Total number of examples after subsampling: {len(X)}")
