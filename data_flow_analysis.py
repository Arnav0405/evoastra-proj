#!/usr/bin/env python3
"""
Data Flow Diagnostic Script
Explains the relationship between images, captions, and training sequences
"""

import pandas as pd
import pickle
import numpy as np

def analyze_data_structure():
    """Analyze and explain the data structure"""
    print("🔍 Image Captioning Data Flow Analysis")
    print("=" * 50)
    
    # Load captions data
    print("1️⃣ Loading caption data...")
    captions_df = pd.read_csv('cap_prep/cleanedcaptions1.csv')
    
    print(f"   Total caption records: {len(captions_df):,}")
    print(f"   Sample records:")
    print(captions_df.head())
    
    # Analyze unique images
    print("\n2️⃣ Analyzing unique images...")
    unique_images = captions_df['image'].unique()
    print(f"   Unique images: {len(unique_images):,}")
    
    # Analyze captions per image
    print("\n3️⃣ Analyzing captions per image...")
    captions_per_image = captions_df.groupby('image').size()
    print(f"   Min captions per image: {captions_per_image.min()}")
    print(f"   Max captions per image: {captions_per_image.max()}")
    print(f"   Average captions per image: {captions_per_image.mean():.2f}")
    
    # Show example of one image with all its captions
    print("\n4️⃣ Example: One image with all its captions...")
    sample_image = unique_images[0]
    sample_captions = captions_df[captions_df['image'] == sample_image]['caption'].tolist()
    print(f"   Image: {sample_image}")
    for i, caption in enumerate(sample_captions, 1):
        print(f"   Caption {i}: {caption}")
    
    # Load tokenizer and analyze sequence creation
    print("\n5️⃣ Analyzing sequence creation...")
    with open('cap_prep/tokenizer1.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Simulate sequence creation for one caption
    sample_caption = sample_captions[0]
    tokens = tokenizer.texts_to_sequences([sample_caption])[0]
    
    print(f"   Sample caption: '{sample_caption}'")
    print(f"   Tokenized: {tokens}")
    
    # Show how training sequences are created
    start_token = 1000  # placeholder
    end_token = 1001    # placeholder
    full_sequence = [start_token] + tokens + [end_token]
    
    print(f"   With special tokens: {full_sequence}")
    print(f"   Training pairs created:")
    
    for i in range(1, len(full_sequence)):
        input_seq = full_sequence[:i]
        target = full_sequence[i]
        print(f"      Input: {input_seq} → Target: {target}")
    
    # Calculate total training examples
    print("\n6️⃣ Calculating total training examples...")
    
    total_examples = 0
    caption_lengths = []
    
    for _, row in captions_df.head(1000).iterrows():  # Sample for speed
        caption = row['caption']
        tokens = tokenizer.texts_to_sequences([caption])[0]
        if len(tokens) > 0:
            sequence_length = len(tokens) + 2  # +2 for start and end tokens
            examples_from_caption = sequence_length - 1  # Number of training pairs
            total_examples += examples_from_caption
            caption_lengths.append(sequence_length)
    
    avg_length = np.mean(caption_lengths)
    avg_examples_per_caption = avg_length - 1
    
    print(f"   Average caption length (with tokens): {avg_length:.1f}")
    print(f"   Average training examples per caption: {avg_examples_per_caption:.1f}")
    
    # Extrapolate to full dataset
    total_captions = len(captions_df)
    estimated_total_examples = total_captions * avg_examples_per_caption
    
    print(f"   Total captions: {total_captions:,}")
    print(f"   Estimated total training examples: {estimated_total_examples:,.0f}")
    
    # Explain the indexing issue
    print("\n7️⃣ Explaining the indexing issue...")
    print("   🔴 THE PROBLEM:")
    print("   - You have 39,874 unique images")
    print("   - Each image has ~5 captions = ~199,370 total captions")
    print("   - Each caption creates ~10 training sequences")
    print("   - Total training sequences: ~2,000,000")
    print("")
    print("   ❌ WRONG APPROACH:")
    print("   - Try to map 39,874 image features to 2,000,000 sequences")
    print("   - Index out of bounds error!")
    print("")
    print("   ✅ CORRECT APPROACH:")
    print("   - Create mapping: image_name → image_features")
    print("   - For each training sequence, look up its image's features")
    print("   - Each sequence gets the features of its corresponding image")
    
    # Show correct data flow
    print("\n8️⃣ Correct data flow:")
    print("   Step 1: Load 39,874 image features into a dictionary")
    print("          image_features = {'image1.jpg': [feature_vector], ...}")
    print("")
    print("   Step 2: Create training sequences with image names")
    print("          sequences = [(image_name, input_seq, target_word), ...]")
    print("")
    print("   Step 3: Map features to sequences")
    print("          for each sequence:")
    print("              feature = image_features[sequence.image_name]")
    print("")
    print("   Result: Each training sequence has its corresponding image features")
    
    print("\n✅ Analysis complete! The fixed script handles this correctly.")

if __name__ == "__main__":
    analyze_data_structure()
