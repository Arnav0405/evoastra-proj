# LSTM Decoder Input/Output Sequence Generation Guide

## Overview

This document explains the process of creating input and output sequences for training an LSTM decoder in an image captioning model.

## Generated Files

- `train_input_sequences.npy`: Input sequences for the decoder (shape: 2,593,832 × 79)
- `train_output_words.npy`: Target words for each input sequence (shape: 2,593,832)
- `sequence_metadata.pkl`: Metadata about the generation process

## Process Explanation

### 1. Sequence Creation Strategy

For each caption, we create multiple training examples using a sliding window approach:

**Original Caption**: "A man in blue shirt"
**Tokenized**: [45, 123, 67, 89, 456] (example token IDs)
**With Special Tokens**: [<start>, 45, 123, 67, 89, 456, <end>]

**Training Examples Created**:

```
Input: [<start>]                    → Target: 45 (A)
Input: [<start>, 45]                → Target: 123 (man)
Input: [<start>, 45, 123]           → Target: 67 (in)
Input: [<start>, 45, 123, 67]       → Target: 89 (blue)
Input: [<start>, 45, 123, 67, 89]   → Target: 456 (shirt)
Input: [<start>, 45, 123, 67, 89, 456] → Target: <end>
```

### 2. Key Components

#### Special Tokens

- `<start>` token (ID: 18363): Marks the beginning of sequence generation
- `<end>` token (ID: 18364): Marks the end of caption generation
- Vocabulary size: 18,365 (including special tokens)

#### Sequence Padding

- Maximum sequence length: 79 tokens
- Input sequences are left-padded with zeros
- This ensures uniform input size for batch processing

### 3. Data Statistics

- **Total training examples**: 2,593,832
- **Input sequence shape**: (2,593,832, 79)
- **Output sequence shape**: (2,593,832,)
- **Sequence length range**: 1 to 79 tokens

## How to Use These Sequences

### Loading the Data

```python
import numpy as np
import pickle

# Load sequences
input_sequences = np.load('cap_prep/train_input_sequences.npy')
output_words = np.load('cap_prep/train_output_words.npy')

# Load metadata
with open('cap_prep/sequence_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

vocab_size = metadata['vocab_size']
max_length = metadata['max_sequence_length']
start_token = metadata['start_token']
end_token = metadata['end_token']
```

### LSTM Model Architecture

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# Convert output words to one-hot encoding
output_categorical = to_categorical(output_words, num_classes=vocab_size)

# Model architecture
decoder_input = Input(shape=(max_length,))
decoder_embedding = Embedding(vocab_size, 256, mask_zero=True)(decoder_input)
decoder_lstm = LSTM(512, return_sequences=False)(decoder_embedding)
decoder_output = Dense(vocab_size, activation='softmax')(decoder_lstm)

model = Model(decoder_input, decoder_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(input_sequences, output_categorical, batch_size=128, epochs=10)
```

### For Image Captioning Integration

In a full image captioning model, you would:

1. **Encoder**: Extract image features using CNN (ResNet, VGG, etc.)
2. **Decoder**: Use these input/output sequences with image features as initial hidden state
3. **Attention**: Optionally add attention mechanism to focus on image regions

```python
# Example integration
image_features = encoder_model(images)  # Shape: (batch_size, feature_dim)
decoder_hidden_state = Dense(512)(image_features)  # Initialize LSTM hidden state

# Use image features to initialize the decoder
decoder_lstm = LSTM(512, return_sequences=False)(
    decoder_embedding,
    initial_state=[decoder_hidden_state, decoder_hidden_state]
)
```

## Training Tips

1. **Batch Size**: Use batch sizes of 64-256 depending on GPU memory
2. **Learning Rate**: Start with 0.001, reduce if loss plateaus
3. **Validation Split**: Use 10-20% of data for validation
4. **Early Stopping**: Monitor validation loss to prevent overfitting
5. **Teacher Forcing**: During training, use ground truth previous words as input

## File Structure

```
cap_prep/
├── train_input_sequences.npy     # Input sequences (decoder input)
├── train_output_words.npy        # Target words (decoder output)
├── sequence_metadata.pkl         # Generation metadata
├── tokenizer1.pkl               # Tokenizer for word-to-index mapping
└── cleanedcaptions1.csv         # Original cleaned captions
```

## Next Steps

1. Load and prepare image features
2. Create the full encoder-decoder model
3. Implement training loop with proper batching
4. Add inference code for generating captions
5. Implement beam search for better caption generation
