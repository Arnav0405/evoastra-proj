import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd # Useful if your data is in CSV

# Load the csv file
df = pd.read_csv('data_extract_code/cleanedcaptions.csv')
raw_data_pairs = [(row['image'], row['caption']) for index, row in df.iterrows()]

all_raw_captions = [caption for img_name, caption in raw_data_pairs]

# Add <start> and <end> tokens to each caption
processed_captions_for_tokenizer_fit = []
for caption in all_raw_captions:
    processed_captions_for_tokenizer_fit.append(f"<start> {caption} <end>")

print(f"Total raw captions: {len(all_raw_captions)}")
print(f"Total processed captions for tokenizer fit: {len(processed_captions_for_tokenizer_fit)}")
print(f"Example processed caption: {processed_captions_for_tokenizer_fit[0]}")


try:
    with open('data_extract_code/tokenizer02.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Loaded existing tokenizer. Re-fitting to include <start> and <end>.")
except FileNotFoundError:
    print("No existing tokenizer found. Initializing a new one.")
    # Initialize a new Tokenizer
    tokenizer = Tokenizer()

# Fit the tokenizer on ALL processed captions (including <start>/<end>)
# This ensures all unique words, including our special tokens, get an ID.
tokenizer.fit_on_texts(processed_captions_for_tokenizer_fit)

# Get the vocabulary size (important for your Embedding layer)
vocab_size = len(tokenizer.word_index) + 1 # +1 because word indices are 1-based, 0 is for padding
print(f"Vocabulary size after fitting: {vocab_size}")

# Get the IDs for your special tokens
start_token_id = tokenizer.word_index.get('<start>')
end_token_id = tokenizer.word_index.get('<end>')

def ensure_special_tokens(tokenizer, tokens):
    max_index = max(tokenizer.word_index.values()) if tokenizer.word_index else 0
    for token in tokens:
        if token not in tokenizer.word_index:
            max_index += 1
            tokenizer.word_index[token] = max_index
            tokenizer.index_word[max_index] = token

if start_token_id is None or end_token_id is None:
    ensure_special_tokens(tokenizer, ['<start>', '<end>'])
    start_token_id = tokenizer.word_index.get('<start>')
    end_token_id = tokenizer.word_index.get('<end>')
else:
    print(f"<start> token ID: {start_token_id}")
    print(f"<end> token ID: {end_token_id}")


with open('data_extract_code/resized_features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# Lists to store the aligned data
aligned_image_features = [] # Will contain image feature vectors (repeated for multiple captions)
aligned_padded_captions = [] # Will contain the numerical padded sequences for each caption

for img_name, caption in raw_data_pairs:
    # 1. Get the image feature vector for the current image
    img_feature = features_dict.get(img_name)
    if img_feature is None:
        print(f"Warning: Image features for {img_name} not found. Skipping this entry.")
        continue

    # 2. Add <start> and <end> to the current caption
    caption_with_tokens = f"<start> {caption} <end>"

    # 3. Convert the caption to a sequence of integers
    sequence = tokenizer.texts_to_sequences([caption_with_tokens])[0] # texts_to_sequences returns a list of lists

    # Add the image feature and sequence to our aligned lists
    aligned_image_features.append(img_feature)
    aligned_padded_captions.append(sequence)

# Determine the maximum sequence length among all newly created sequences
# This ensures consistent padding for all captions
max_caption_length = max(len(seq) for seq in aligned_padded_captions)
print(f"\nMax caption length (including <start> and <end> across all captions): {max_caption_length}")

# Pad all caption sequences to the determined max_caption_length
# 'post' padding is common for image captioning tasks
final_padded_sequences = pad_sequences(aligned_padded_captions, maxlen=max_caption_length, padding='post')

# Convert aligned_image_features to a NumPy array
final_image_features_array = np.array(aligned_image_features)

print(f"Shape of final_image_features_array: {final_image_features_array.shape}")
print(f"Shape of final_padded_sequences: {final_padded_sequences.shape}")

# Example of an aligned pair:
print(f"\nFirst image feature shape: {final_image_features_array[0].shape}")
print(f"First padded caption sequence (numerical): {final_padded_sequences[0]}")


# Input for the language model part of your decoder (shifted right by one)
decoder_input_data = final_padded_sequences[:, :-1]

# Target output for the language model part (the next word in the sequence)
decoder_target_data = final_padded_sequences[:, 1:]

print(f"\nShape of decoder_input_data: {decoder_input_data.shape}")
print(f"Shape of decoder_target_data: {decoder_target_data.shape}")

# Example of an input-output pair for the first caption:
print("\nFirst Caption Input Sequence (numerical):")
print(decoder_input_data[0])

print("First Caption Target Sequence (numerical):")
print(decoder_target_data[0])

# Save the updated tokenizer (it now knows about <start> and <end>)
with open('./data_extract_code/saved/tokenizer_final.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Saved tokenizer_final.pkl")

# Save the final aligned image features array
np.save('data_extract_code/saved/aligned_image_features.npy', final_image_features_array)
print("Saved aligned_image_features.npy")

# Save the decoder input and target sequences
np.save('./data_extract_code/saved/decoder_input_data.npy', decoder_input_data)
np.save('./data_extract_code/saved/decoder_target_data.npy', decoder_target_data)
print("Saved decoder_input_data.npy and decoder_target_data.npy")

# Optional: Save the max_caption_length if you need it later for model architecture
with open('./data_extract_code/saved/max_caption_length.pkl', 'wb') as f:
    pickle.dump(max_caption_length, f)
print("Saved max_caption_length.pkl")