import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Step 1: Load captions from .txt file
with open('captions_only.txt', 'r', encoding='utf-8') as f:
    captions = f.read().strip().split('\n')

print(f"✅ Loaded {len(captions)} captions.")

# Step 2: Tokenize the captions
tokenizer = Tokenizer(oov_token='<unk>')
tokenizer.fit_on_texts(captions)
sequences = tokenizer.texts_to_sequences(captions)

# Step 3: Set max caption length
max_length = max(len(seq) for seq in sequences)
print("📏 Max caption length:", max_length)

# Step 4: Pad the sequences
padded_captions = pad_sequences(sequences, maxlen=max_length, padding='post')

# Step 5: Save for later use
np.save('padded_captions.npy', padded_captions)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Output sample
print("🧾 Shape of padded captions:", padded_captions.shape)
print("🧱 Sample padded caption:", padded_captions[0])
