from itertools import product
from treelib import Node, Tree
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import io
from contextlib import redirect_stdout

# Initialize the tokenizer and model from the HuggingFace Transformers library.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Define the input sequence.
input_sequence = "It is important for all countries to try harder to reduce carbon emissions because"

# Encode the input text.
input_ids = tokenizer(input_sequence, return_tensors="pt").input_ids

# Define arrays for different temperature and top_p values
temperature_values = [0.2, 0.5, 0.7, 1.0]
top_p_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# Dictionary to store trees for each combination of temperature and top_p
trees = {}

# Loop over each combination of temperature and top_p values
for temperature, top_p in product(temperature_values, top_p_values):
    # Generate text using the model with the current combination of temperature and top_p
    outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=30,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True
    )

    # Retrieve the scores (logits) for each token generated
    generated_scores = outputs['scores']

    # Create a new tree for this combination
    tree = Tree()
    tree.create_node(f"Root (Temp: {temperature}, Top-p: {top_p})", "root")  # Root node with params

    for i, score in enumerate(generated_scores):
        parent_id = f"step_{i-1}_option_0" if i > 0 else "root"

        # Apply softmax to convert logits to probabilities
        probs = torch.softmax(score, dim=-1)

        # Get the top 3 probabilities and their token indices
        top_probs, top_indices = torch.topk(probs, 3, dim=-1)

        for j in range(top_indices.size(1)):
            token_id = top_indices[0][j].item()
            prob = top_probs[0][j].item()
            token = tokenizer.decode([token_id])  # Decode the list of one token ID

            node_id = f"step_{i}_option_{j}"
            node_label = f"{token} ({prob:.2f})"
            tree.create_node(node_label, node_id, parent=parent_id)

    # Save the tree for this combination
    trees[(temperature, top_p)] = tree
    tree.save2file("tree-"+str(temperature)+"-"+str(top_p)+".txt")
