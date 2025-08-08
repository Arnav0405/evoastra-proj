import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import pandas as pd

# Load the tokenizer from file
with open('data_extract_code\\tokenizer02.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# View the number of words in the vocabulary
print(f"🔢 Vocabulary size: {len(tokenizer.word_index)}")
print(tokenizer.word_index[''])  # Check padding token index
# View a few sample word-index mappings
print("📚 Sample word_index (word → index):")
mappings = {}

# for word, index in list(tokenizer.word_index.items())[:-1]:
#     mappings[word] = index
#     print(f"{word} → {index}")

# Save the mappings to a CSV file
mappings_df = pd.DataFrame(list(mappings.items()), columns=['word', 'index'])
# mappings_df.to_csv('data_extract_code\\word_index_mappings.csv', index=False)

# View index-word mapping (reverse lookup)
index_word = {index: word for word, index in tokenizer.word_index.items()}

# print("\n🔁 Sample index_word (index → word):")
# for i in range(1, 21):  # skip 0 (usually padding)
    # print(f"{i} → {index_word.get(i, '<unk>')}")
