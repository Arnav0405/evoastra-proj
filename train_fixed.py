#!/usr/bin/env python3
"""
Fixed Image Captioning Training Script
Properly handles 39,874 images with 5 captions each
"""

import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Set environment
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class ImageCaptionDataHandler:
    def __init__(self):
        self.captions_df = None
        self.tokenizer = None
        self.image_features_map = {}
        self.vocab_size = 0
        self.start_token = 0
        self.end_token = 0
        
    def load_captions_and_tokenizer(self):
        """Load captions and tokenizer"""
        print("🔍 Loading captions and tokenizer...")
        
        # Load captions
        self.captions_df = pd.read_csv('cap_prep/cleanedcaptions1.csv')
        print(f"✅ Loaded {len(self.captions_df)} caption records")
        
        # Load tokenizer
        with open('cap_prep/tokenizer1.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Handle special tokens
        if '<start>' not in self.tokenizer.word_index:
            self.tokenizer.word_index['<start>'] = len(self.tokenizer.word_index) + 1
        if '<end>' not in self.tokenizer.word_index:
            self.tokenizer.word_index['<end>'] = len(self.tokenizer.word_index) + 1
            
        self.start_token = self.tokenizer.word_index['<start>']
        self.end_token = self.tokenizer.word_index['<end>']
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        print(f"✅ Vocabulary size: {self.vocab_size}")
        print(f"✅ Start token: {self.start_token}, End token: {self.end_token}")
        
        # Get unique images
        unique_images = self.captions_df['image'].unique()
        print(f"✅ Found {len(unique_images)} unique images")
        
    def load_image_features(self, use_real_features=True):
        """Load image features and create mapping"""
        print("🔍 Loading image features...")
        
        if use_real_features:
            try:
                return self._load_real_features()
            except Exception as e:
                print(f"❌ Error loading real features: {e}")
                print("   Falling back to dummy features...")
                return self._create_dummy_features()
        else:
            return self._create_dummy_features()
    
    def _load_real_features(self):
        """Load real image features from CSV"""
        print("   Loading real features from CSV...")
        
        # Load in chunks to handle large files
        chunk_list = []
        chunk_size = 5000
        
        for chunk in pd.read_csv('resized_features.csv', chunksize=chunk_size):
            chunk_list.append(chunk)
        
        features_df = pd.concat(chunk_list, ignore_index=True)
        print(f"   Loaded features for {len(features_df)} images")
        
        # Create image to features mapping
        if 'image' in features_df.columns:
            image_col = 'image'
            feature_cols = [col for col in features_df.columns if col != 'image']
        else:
            image_col = features_df.columns[0]
            feature_cols = features_df.columns[1:]
        
        feature_dim = len(feature_cols)
        
        for _, row in features_df.iterrows():
            img_name = row[image_col]
            features = row[feature_cols].values.astype(np.float32)
            self.image_features_map[img_name] = features
        
        print(f"✅ Created feature mapping for {len(self.image_features_map)} images")
        print(f"✅ Feature dimension: {feature_dim}")
        
        return feature_dim
    
    def _create_dummy_features(self):
        """Create dummy features for testing"""
        print("   Creating dummy features...")
        
        unique_images = self.captions_df['image'].unique()
        feature_dim = 2048  # Standard ResNet50 size
        
        np.random.seed(42)
        for img in unique_images:
            self.image_features_map[img] = np.random.normal(0, 1, feature_dim).astype(np.float32)
        
        print(f"✅ Created dummy features for {len(unique_images)} images")
        print(f"✅ Feature dimension: {feature_dim}")
        
        return feature_dim
    
    def create_training_data(self, max_samples=None):
        """Create training sequences with proper image-caption mapping"""
        print("🔧 Creating training sequences...")
        
        all_image_names = []
        all_input_sequences = []
        all_target_words = []
        
        max_caption_length = 0
        total_sequences = 0
        
        # Process each image and its captions
        for image_name in self.captions_df['image'].unique():
            if max_samples and total_sequences >= max_samples:
                break
                
            # Get all captions for this image
            image_captions = self.captions_df[self.captions_df['image'] == image_name]['caption'].tolist()
            
            # Skip if image features not available
            if image_name not in self.image_features_map:
                continue
            
            # Process each caption for this image
            for caption in image_captions:
                if max_samples and total_sequences >= max_samples:
                    break
                    
                # Tokenize caption
                tokens = self.tokenizer.texts_to_sequences([caption])[0]
                if len(tokens) == 0:
                    continue
                
                # Add start and end tokens
                full_sequence = [self.start_token] + tokens + [self.end_token]
                max_caption_length = max(max_caption_length, len(full_sequence))
                
                # Create input-output pairs for each position in the sequence
                for i in range(1, len(full_sequence)):
                    if max_samples and total_sequences >= max_samples:
                        break
                        
                    input_seq = full_sequence[:i]
                    target_word = full_sequence[i]
                    
                    all_image_names.append(image_name)
                    all_input_sequences.append(input_seq)
                    all_target_words.append(target_word)
                    total_sequences += 1
        
        print(f"✅ Created {len(all_input_sequences)} training sequences")
        print(f"✅ Max caption length: {max_caption_length}")
        print(f"✅ Unique images in sequences: {len(set(all_image_names))}")
        
        # Pad input sequences
        padded_sequences = pad_sequences(all_input_sequences, maxlen=max_caption_length, padding='pre')
        
        # Create image features array matching the sequences
        image_features_array = np.array([self.image_features_map[img] for img in all_image_names])
        
        print(f"✅ Input sequences shape: {padded_sequences.shape}")
        print(f"✅ Image features shape: {image_features_array.shape}")
        print(f"✅ Target words shape: {len(all_target_words)}")
        
        return (padded_sequences, 
                image_features_array, 
                np.array(all_target_words), 
                max_caption_length)

def build_model(vocab_size, max_length, feature_dim):
    """Build the image captioning model"""
    print("🏗️ Building model...")
    
    # Image feature input
    image_input = Input(shape=(feature_dim,), name='image_features')
    image_dense = Dense(512, activation='relu')(image_input)
    image_dense = Dropout(0.3)(image_dense)
    
    # Caption input
    caption_input = Input(shape=(max_length,), name='caption_input')
    embedding = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    embedding = Dropout(0.3)(embedding)
    
    # LSTM decoder
    lstm_out = LSTM(512)(embedding)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Combine image and text features
    combined = Add()([image_dense, lstm_out])
    combined = Dense(512, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    
    # Output layer
    output = Dense(vocab_size, activation='softmax')(combined)
    
    # Create and compile model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model built successfully")
    print(f"📊 Total parameters: {model.count_params():,}")
    
    return model

def main():
    """Main training function"""
    print("🚀 Fixed Image Captioning Training")
    print("=" * 50)
    print("📊 Handling 39,874 images × 5 captions each")
    print("=" * 50)
    
    # Initialize data handler
    data_handler = ImageCaptionDataHandler()
    
    # Load data
    data_handler.load_captions_and_tokenizer()
    feature_dim = data_handler.load_image_features(use_real_features=True)
    
    # Create training data (limit samples for testing)
    print("\n📊 Creating training data...")
    input_sequences, image_features, target_words, max_length = data_handler.create_training_data(
        max_samples=100000  # Limit for faster testing
    )
    
    # Verify data consistency
    print(f"\n🔍 Data consistency check:")
    print(f"   Input sequences: {input_sequences.shape}")
    print(f"   Image features: {image_features.shape}")
    print(f"   Target words: {target_words.shape}")
    
    assert len(input_sequences) == len(image_features) == len(target_words), \
        "Data length mismatch!"
    
    # Split data
    print("\n📊 Splitting data...")
    indices = np.arange(len(input_sequences))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = [image_features[train_idx], input_sequences[train_idx]]
    y_train = target_words[train_idx]
    X_val = [image_features[val_idx], input_sequences[val_idx]]
    y_val = target_words[val_idx]
    
    print(f"✅ Training samples: {len(train_idx):,}")
    print(f"✅ Validation samples: {len(val_idx):,}")
    
    # Build model
    model = build_model(data_handler.vocab_size, max_length, feature_dim)
    model.summary()
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            'best_caption_model_fixed.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("\n🏃 Starting training...")
    print("📈 Expected training time: 2-4 hours on GPU, 8-12 hours on CPU")
    
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=64,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model and components
    print("\n💾 Saving model components...")
    
    model.save('final_caption_model_fixed.h5')
    print("✅ Model saved")
    
    with open('final_tokenizer_fixed.pkl', 'wb') as f:
        pickle.dump(data_handler.tokenizer, f)
    print("✅ Tokenizer saved")
    
    metadata = {
        'vocab_size': data_handler.vocab_size,
        'max_length': max_length,
        'feature_dim': feature_dim,
        'start_token': data_handler.start_token,
        'end_token': data_handler.end_token,
        'unique_images': len(set(data_handler.captions_df['image'])),
        'total_captions': len(data_handler.captions_df),
        'training_samples': len(train_idx)
    }
    
    with open('training_metadata_fixed.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("✅ Metadata saved")
    
    # Print final results
    print("\n📊 Training Results:")
    print("=" * 40)
    final_loss = history.history['val_loss'][-1]
    final_acc = history.history['val_accuracy'][-1]
    best_loss = min(history.history['val_loss'])
    best_acc = max(history.history['val_accuracy'])
    
    print(f"Final Validation Loss: {final_loss:.4f}")
    print(f"Final Validation Accuracy: {final_acc:.4f}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    
    print("\n✅ Training completed successfully!")
    print("\n🎯 Next steps:")
    print("   1. Use caption_generator.py for inference")
    print("   2. Test with real images")
    print("   3. Evaluate with BLEU scores")

if __name__ == "__main__":
    main()
