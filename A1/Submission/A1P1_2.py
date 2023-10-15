import torch
import torchtext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=100)    # embedding size = 50

def print_closest_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

def print_closest_cosine_words(vec, n=5):
    # Compute the cosine similarity between the input vector and all the words
    similarities = cosine_similarity(vec.reshape(1, -1), glove.vectors)
    
    # find the n most similar words
    top_indices = np.argsort(similarities[0])[::-1][:n]

    # print the most similar words and their similarities
    for idx in top_indices:
        similarity = similarities[0][idx]
        print(glove.itos[idx], "\t%5.2f" % similarity)

def compare_word_similarities(word):
    print(f"compare '{word}' word similar to:")
    print("\ncosine: \nword\tcosine similarity")
    print_closest_cosine_words(glove[word], n=10)
    
    print("\nEuclidean: \nword\tEuclidean distance")
    print_closest_words(glove[word], n=10)

if __name__ == '__main__':
    compare_word_similarities('dog')
    print("------------------------------------")
    compare_word_similarities('computer')