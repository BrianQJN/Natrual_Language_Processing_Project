import torchtext
import A1P1_2

glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=300)    # embedding size = 50
# Generate the second word in the relationship
relation = glove['king'] - glove['prince'] + glove['princess']
A1P1_2.print_closest_cosine_words(relation)
print("\n")

# Generate 10 more examples of the same relationship from 10 other words
examples = [
    'emperor', 'monarch', 'tsar', 'sultan', 'czar', 'duke', 'lord', 'nobleman', 'ruler', 'regent'
]

for example in examples:
    word_vector = glove[example]
    predicted_word = word_vector + relation
    print(f"{example} + Relationship = ")
    A1P1_2.print_closest_cosine_words(predicted_word)
    print("\n")