#!/usr/bin/env python3
"""
Sequence Verification Script
This script demonstrates how the generated input/output sequences work
"""

import numpy as np
import pickle

def load_sequences():
    """Load the generated sequences and metadata"""
    print("🔍 Loading generated sequences...")
    
    # Load sequences
    input_sequences = np.load('cap_prep/train_input_sequences.npy')
    output_words = np.load('cap_prep/train_output_words.npy')
    
    # Load metadata
    with open('cap_prep/sequence_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Load tokenizer for decoding
    with open('cap_prep/tokenizer1.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    return input_sequences, output_words, metadata, tokenizer

def create_index_to_word_mapping(tokenizer):
    """Create mapping from token indices to words"""
    return {v: k for k, v in tokenizer.word_index.items()}

def display_sequence_examples(input_sequences, output_words, index_to_word, num_examples=5):
    """Display examples of input-output pairs"""
    print(f"\n📊 Displaying {num_examples} sequence examples:")
    print("=" * 80)
    
    for i in range(num_examples):
        input_seq = input_sequences[i]
        output_word = output_words[i]
        
        # Remove padding (zeros) from input sequence
        input_tokens = input_seq[input_seq != 0]
        
        # Decode input sequence
        input_words = [index_to_word.get(token, f"<UNK:{token}>") for token in input_tokens]
        output_word_text = index_to_word.get(output_word, f"<UNK:{output_word}>")
        
        print(f"\nExample {i+1}:")
        print(f"  Input Sequence:  {' '.join(input_words)}")
        print(f"  Target Word:     {output_word_text}")
        print(f"  Input Length:    {len(input_tokens)} tokens")
        print(f"  Input IDs:       {input_tokens.tolist()}")
        print(f"  Target ID:       {output_word}")

def analyze_sequence_statistics(input_sequences, output_words, metadata):
    """Analyze and display sequence statistics"""
    print("\n📈 Sequence Statistics:")
    print("=" * 50)
    
    print(f"Total training examples: {metadata['total_examples']:,}")
    print(f"Vocabulary size: {metadata['vocab_size']:,}")
    print(f"Maximum sequence length: {metadata['max_sequence_length']}")
    print(f"Start token ID: {metadata['start_token']}")
    print(f"End token ID: {metadata['end_token']}")
    
    # Analyze input sequence lengths (without padding)
    actual_lengths = []
    for seq in input_sequences[:10000]:  # Sample for efficiency
        actual_length = len(seq[seq != 0])
        actual_lengths.append(actual_length)
    
    print(f"\nInput Sequence Length Analysis (sample of 10,000):")
    print(f"  Minimum length: {min(actual_lengths)}")
    print(f"  Maximum length: {max(actual_lengths)}")
    print(f"  Average length: {np.mean(actual_lengths):.2f}")
    
    # Analyze output word distribution
    unique_words, counts = np.unique(output_words[:10000], return_counts=True)
    print(f"\nOutput Word Distribution (sample of 10,000):")
    print(f"  Unique words in sample: {len(unique_words)}")
    print(f"  Most frequent word ID: {unique_words[np.argmax(counts)]} (appears {max(counts)} times)")

def demonstrate_training_batch(input_sequences, output_words, metadata, batch_size=3):
    """Demonstrate how to create training batches"""
    print(f"\n🎯 Training Batch Example (batch_size={batch_size}):")
    print("=" * 60)
    
    # Create a small batch
    batch_inputs = input_sequences[:batch_size]
    batch_outputs = output_words[:batch_size]
    
    print(f"Batch input shape: {batch_inputs.shape}")
    print(f"Batch output shape: {batch_outputs.shape}")
    
    print("\nFor LSTM training, you would:")
    print("1. Convert outputs to one-hot encoding:")
    print(f"   output_categorical = to_categorical(batch_outputs, num_classes={metadata['vocab_size']})")
    print("2. Feed batch_inputs to LSTM decoder")
    print("3. Compare LSTM predictions with output_categorical")

def main():
    """Main demonstration function"""
    print("🚀 LSTM Sequence Verification")
    print("=" * 50)
    
    # Load data
    input_sequences, output_words, metadata, tokenizer = load_sequences()
    index_to_word = create_index_to_word_mapping(tokenizer)
    
    # Display statistics
    analyze_sequence_statistics(input_sequences, output_words, metadata)
    
    # Show examples
    display_sequence_examples(input_sequences, output_words, index_to_word)
    
    # Demonstrate training batch
    demonstrate_training_batch(input_sequences, output_words, metadata)
    
    print("\n✅ Verification complete!")
    print("\nThese sequences are ready for LSTM decoder training.")
    print("See SEQUENCE_GENERATION_GUIDE.md for detailed usage instructions.")

if __name__ == "__main__":
    main()
