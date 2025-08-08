import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle

# Load the tokenizer from file
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# View the number of words in the vocabulary
print(f"🔢 Vocabulary size: {len(tokenizer.word_index)}")

# View a few sample word-index mappings
print("📚 Sample word_index (word → index):")
for word, index in list(tokenizer.word_index.items())[:10]:
    print(f"{word} → {index}")

# View index-word mapping (reverse lookup)
index_word = {index: word for word, index in tokenizer.word_index.items()}

print("\n🔁 Sample index_word (index → word):")
for i in range(1, 21):  # skip 0 (usually padding)
    print(f"{i} → {index_word.get(i, '<unk>')}")
