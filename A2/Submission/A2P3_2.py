import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('Submission/data.tsv', sep='\t')

# Splitting the data ensuring equal class representation

# Separate positive and negative samples
positive_data = data[data['label'] == 1]
negative_data = data[data['label'] == 0]

# Split positive data
train_pos, temp_pos = train_test_split(positive_data, test_size=0.36, random_state=42)
valid_pos, test_pos = train_test_split(temp_pos, test_size=0.5556, random_state=42)

# Split negative data
train_neg, temp_neg = train_test_split(negative_data, test_size=0.36, random_state=42)
valid_neg, test_neg = train_test_split(temp_neg, test_size=0.5556, random_state=42)

# Combine positive and negative samples for each split
train_data = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)  # Shuffle the data
valid_data = pd.concat([valid_pos, valid_neg]).sample(frac=1, random_state=42)
test_data = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)
overfit_data = pd.concat([positive_data.sample(25,random_state=42), negative_data.sample(25,random_state=42)]).sample(frac=1, random_state=42)

# Verify the proportions and equal class representation
proportions = (len(train_data) / len(data), len(valid_data) / len(data), len(test_data) / len(data))
class_balance_train = train_data['label'].value_counts().to_dict()
class_balance_valid = valid_data['label'].value_counts().to_dict()
class_balance_test = test_data['label'].value_counts().to_dict()
class_balance_overfit = overfit_data['label'].value_counts().to_dict()

# Save the datasets to tsv files
train_data.to_csv('data/train.tsv', sep='\t', index=False)
valid_data.to_csv('data/validation.tsv', sep='\t', index=False)
test_data.to_csv('data/test.tsv', sep='\t', index=False)
overfit_data.to_csv('data/overfit.tsv', sep='\t', index=False)

print(proportions, class_balance_train, class_balance_valid, class_balance_test, class_balance_overfit)