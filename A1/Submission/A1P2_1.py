import torch
import torchtext.vocab as vocab
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Load GloVe vectors with embedding dimension 50
glove = vocab.GloVe(name="6B", dim=50)

def compare_words_to_category(category_words, word_to_measure):
    # Initialize similarity lists
    similarities = []
    avg_embedding = torch.zeros(50)  # Initialize the average embedding

    # Calculate cosine similarity between the word and each word in the category
    for category_word in category_words:
        if category_word in glove.stoi and word_to_measure in glove.stoi:
            # Calculate cosine similarity
            similarity = cosine_similarity(glove[word_to_measure].reshape(1, -1), glove[category_word].reshape(1, -1))[0][0]
            similarities.append(similarity)

            # Add the word's embedding to the average embedding
            avg_embedding += glove[category_word]

    # Calculate the average cosine similarity
    avg_cosine_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    # Calculate the average embedding
    if len(category_words) > 0:
        avg_embedding /= len(category_words)

    # Calculate cosine similarity between the word and the average embedding
    avg_embedding_cosine_similarity = cosine_similarity(glove[word_to_measure].reshape(1, -1), avg_embedding.reshape(1, -1))[0][0]

    return avg_cosine_similarity, avg_embedding_cosine_similarity


if __name__ == '__main__':
    # # Define the color meaning category words
    # color_category = ["color", "red", "green", "blue", "yellow"]

    # # Words to measure
    # words_to_measure = ["greenhouse", "sky", "grass", "azure", "scissors", "microphone", "president"]

    # # Create a table header
    # print("Word".ljust(15), "Method (a)".ljust(15), "Method (b)".ljust(15))

    # # Calculate and print the similarity for each word
    # for word in words_to_measure:
    #     avg_cosine_similarity, avg_embedding_cosine_similarity = compare_words_to_category(color_category, word)
    #     print(word.ljust(15), f"{avg_cosine_similarity:.4f}".ljust(15), f"{avg_embedding_cosine_similarity:.4f}".ljust(15))
    
    # Define the temperature meaning category words
    # temperature_category = ["temperature", "hot", "cold", "warm", "freezing", "boiling", "chilly", "frigid", "scorching", "mild"]

    # # Words to measure
    # words_to_measure_temperature = ["summer", "ice", "thermometer", "oven", "snow", "beach", "jacket", "fire", "coffee", "desert"]

    # # Create a table header
    # print("Word".ljust(15), "Method (a)".ljust(15), "Method (b)".ljust(15))

    # # Calculate and print the similarity for each word
    # for word in words_to_measure_temperature:
    #     avg_cosine_similarity, avg_embedding_cosine_similarity = compare_words_to_category(temperature_category, word)
    #     print(word.ljust(15), f"{avg_cosine_similarity:.4f}".ljust(15), f"{avg_embedding_cosine_similarity:.4f}".ljust(15))

    # Define the color and temperature categories
    color_category = ["color", "red", "green", "blue", "yellow"]
    temperature_category = ["temperature", "hot", "cold", "warm", "freezing", "boiling", "chilly", "frigid", "scorching", "mild"]

    # Words to measure
    words_to_measure = ["sun", "moon", "winter", "rain", "cow", "wrist", "wind", "prefix", "ghost", "glow", "heated", "cool"]

    # Initialize lists to store color and temperature probabilities
    temperature = []
    color = []

    # Calculate probabilities for each word
    for word in words_to_measure:
        avg_cosine_similarity_color_a, avg_cosine_similarity_temp_a = compare_words_to_category(color_category, word)
        avg_cosine_similarity_color_b, avg_cosine_similarity_temp_b = compare_words_to_category(temperature_category, word)
        
        # Apply softmax to the similarity scores for both color and temperature separately
        color.append(max(avg_cosine_similarity_color_a, avg_cosine_similarity_color_b))
        temperature.append(max(avg_cosine_similarity_temp_a, avg_cosine_similarity_temp_b))

    # Convert lists to NumPy arrays for plotting
    color = np.array(color)
    temperature = np.array(temperature)
    

    # Plot words in two dimensions (color vs. temperature)
    plt.figure(figsize=(8, 6))
    plt.scatter(color, temperature, label='Data Points', color='b', marker='o', s=100, alpha=0.7)

    for i, label in enumerate(words_to_measure):
        plt.text(color[i], temperature[i], label, fontsize=10, ha='center', va='bottom')

    plt.xlabel('Color')
    plt.ylabel('Temperature')
    plt.title('Scatter Plot of Color vs. Temperature')

    plt.legend()
    plt.grid()
    plt.show()