#!/usr/bin/env python3
"""
Caption Generation Inference Script
Use this after training to generate captions for images
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CaptionGenerator:
    def __init__(self, model_path='final_caption_model.h5', 
                 tokenizer_path='final_tokenizer.pkl',
                 metadata_path='training_metadata.pkl'):
        
        print("🔍 Loading model components...")
        
        # Load model
        self.model = load_model(model_path)
        print("✅ Model loaded")
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print("✅ Tokenizer loaded")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print("✅ Metadata loaded")
        
        # Extract metadata
        self.vocab_size = self.metadata['vocab_size']
        self.max_length = self.metadata['max_length']
        self.start_token = self.metadata['start_token']
        self.end_token = self.metadata['end_token']
        self.feature_dim = self.metadata['feature_dim']
        
        # Create index to word mapping
        self.index_to_word = {v: k for k, v in self.tokenizer.word_index.items()}
        
        print(f"📊 Model ready - Vocab: {self.vocab_size}, Max length: {self.max_length}")
    
    def generate_caption(self, image_features, max_words=20, temperature=1.0):
        """
        Generate caption for given image features
        
        Args:
            image_features: numpy array of image features
            max_words: maximum number of words to generate
            temperature: sampling temperature (higher = more random)
        """
        
        # Start with the start token
        caption = [self.start_token]
        
        for _ in range(max_words):
            # Prepare input sequence
            input_seq = pad_sequences([caption], maxlen=self.max_length, padding='pre')
            
            # Predict next word
            predictions = self.model.predict([image_features.reshape(1, -1), input_seq], verbose=0)
            
            # Apply temperature sampling
            if temperature > 0:
                predictions = np.log(predictions + 1e-8) / temperature
                predictions = np.exp(predictions)
                predictions = predictions / np.sum(predictions)
                
                # Sample from distribution
                next_word_idx = np.random.choice(len(predictions[0]), p=predictions[0])
            else:
                # Greedy selection
                next_word_idx = np.argmax(predictions[0])
            
            # Check for end token
            if next_word_idx == self.end_token:
                break
            
            # Add word to caption
            caption.append(next_word_idx)
        
        # Convert to words
        words = []
        for idx in caption[1:]:  # Skip start token
            word = self.index_to_word.get(idx, '<UNK>')
            if word != '<UNK>':
                words.append(word)
        
        return ' '.join(words)
    
    def generate_multiple_captions(self, image_features, num_captions=5, **kwargs):
        """Generate multiple captions for the same image"""
        captions = []
        for i in range(num_captions):
            caption = self.generate_caption(image_features, **kwargs)
            captions.append(f"{i+1}. {caption}")
        return captions
    
    def beam_search_caption(self, image_features, beam_width=3, max_words=20):
        """
        Generate caption using beam search for better quality
        """
        # Initialize beams with start token
        beams = [([self.start_token], 0.0)]  # (sequence, score)
        
        for _ in range(max_words):
            new_beams = []
            
            for sequence, score in beams:
                # Prepare input
                input_seq = pad_sequences([sequence], maxlen=self.max_length, padding='pre')
                
                # Get predictions
                predictions = self.model.predict([image_features.reshape(1, -1), input_seq], verbose=0)
                
                # Get top k predictions
                top_indices = np.argsort(predictions[0])[-beam_width:]
                
                for idx in top_indices:
                    new_score = score + np.log(predictions[0][idx] + 1e-8)
                    new_sequence = sequence + [idx]
                    
                    # Check for end token
                    if idx == self.end_token:
                        new_beams.append((new_sequence, new_score))
                    else:
                        new_beams.append((new_sequence, new_score))
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if all beams ended
            if all(seq[-1] == self.end_token for seq, _ in beams):
                break
        
        # Get best sequence
        best_sequence = beams[0][0]
        
        # Convert to words
        words = []
        for idx in best_sequence[1:]:  # Skip start token
            if idx == self.end_token:
                break
            word = self.index_to_word.get(idx, '<UNK>')
            if word != '<UNK>':
                words.append(word)
        
        return ' '.join(words)

def demo_caption_generation():
    """Demo function to show how to use the caption generator"""
    
    try:
        # Initialize generator
        generator = CaptionGenerator()
        
        print("\n🎯 Caption Generation Demo")
        print("=" * 50)
        
        # Create dummy image features for demo
        dummy_features = np.random.normal(0, 1, generator.feature_dim)
        
        print("\n📝 Generating captions with different methods:")
        
        # Method 1: Simple generation
        print("\n1. Simple Generation:")
        caption = generator.generate_caption(dummy_features)
        print(f"   Caption: {caption}")
        
        # Method 2: Multiple captions
        print("\n2. Multiple Captions:")
        captions = generator.generate_multiple_captions(dummy_features, num_captions=3)
        for cap in captions:
            print(f"   {cap}")
        
        # Method 3: Beam search
        print("\n3. Beam Search:")
        beam_caption = generator.beam_search_caption(dummy_features)
        print(f"   Caption: {beam_caption}")
        
        # Method 4: Temperature sampling
        print("\n4. Creative Generation (high temperature):")
        creative_caption = generator.generate_caption(dummy_features, temperature=1.5)
        print(f"   Caption: {creative_caption}")
        
        print("\n✅ Demo completed!")
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("🔧 Make sure training is completed and model files exist:")
        print("   - final_caption_model.h5")
        print("   - final_tokenizer.pkl") 
        print("   - training_metadata.pkl")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def generate_for_real_image(image_path_or_features):
    """
    Generate caption for a real image
    
    Args:
        image_path_or_features: Either path to image or pre-extracted features
    """
    
    try:
        generator = CaptionGenerator()
        
        # If it's an image path, you'll need to extract features first
        if isinstance(image_path_or_features, str):
            print(f"🖼️ Processing image: {image_path_or_features}")
            # Here you would load the image and extract features using your CNN
            # For now, using dummy features
            image_features = np.random.normal(0, 1, generator.feature_dim)
            print("⚠️  Using dummy features - replace with real CNN feature extraction")
        else:
            image_features = image_path_or_features
        
        # Generate caption
        caption = generator.beam_search_caption(image_features)
        
        print(f"📝 Generated Caption: {caption}")
        return caption
        
    except Exception as e:
        print(f"❌ Error generating caption: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Image Caption Generator")
    print("=" * 40)
    
    # Run demo
    demo_caption_generation()
    
    print("\n💡 Usage:")
    print("   generator = CaptionGenerator()")
    print("   caption = generator.generate_caption(image_features)")
    print("   beam_caption = generator.beam_search_caption(image_features)")
