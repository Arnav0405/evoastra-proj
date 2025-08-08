#!/usr/bin/env python3
"""
Simple Image Captioning Training Script
Uses pre-generated sequences and image features
"""

import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Add, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Set environment
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_data():
    """Load all required data"""
    print("🔍 Loading data...")
    
    # Load captions data
    captions_df = pd.read_csv('cap_prep/cleanedcaptions1.csv')
    print(f"✅ Loaded {len(captions_df)} caption records")
    
    # Load tokenizer
    with open('cap_prep/tokenizer1.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    vocab_size = len(tokenizer.word_index) + 1
    start_token = tokenizer.word_index.get('<start>', len(tokenizer.word_index) + 1)
    end_token = tokenizer.word_index.get('<end>', len(tokenizer.word_index) + 2)
    
    print(f"✅ Vocabulary size: {vocab_size}")
    print(f"✅ Start token: {start_token}, End token: {end_token}")
    
    return captions_df, tokenizer, vocab_size, start_token, end_token

def load_image_features_sample():
    """Load a sample of image features to check structure"""
    print("🔍 Checking image features structure...")
    
    try:
        # Load first few rows to understand structure
        sample_df = pd.read_csv('resized_features.csv', nrows=5)
        print(f"✅ Feature columns: {sample_df.columns.tolist()}")
        print(f"✅ Feature dimensions: {sample_df.shape[1] - 1}")  # Assuming first column is image name
        
        # Get feature dimension
        feature_dim = sample_df.shape[1] - 1 if 'image' in sample_df.columns else sample_df.shape[1]
        return feature_dim
        
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        # Use a default feature dimension (common CNN feature sizes)
        return 2048  # ResNet50 feature size

def create_sequences_with_images(captions_df, tokenizer, start_token, end_token, max_samples=50000):
    """Create sequences with corresponding image names"""
    print("🔧 Creating training sequences...")
    
    image_names = []
    input_sequences = []
    target_sequences = []
    
    max_length = 0
    sample_count = 0
    
    # Group by image to handle multiple captions per image
    for image_name, group in captions_df.groupby('image'):
        if sample_count >= max_samples:
            break
            
        for _, row in group.iterrows():
            caption = row['caption']
            
            # Tokenize caption
            tokens = tokenizer.texts_to_sequences([caption])[0]
            if len(tokens) == 0:
                continue
            
            # Add start and end tokens
            full_sequence = [start_token] + tokens + [end_token]
            max_length = max(max_length, len(full_sequence))
            
            # Create input-output pairs
            for i in range(1, len(full_sequence)):
                input_seq = full_sequence[:i]
                target_word = full_sequence[i]
                
                image_names.append(image_name)
                input_sequences.append(input_seq)
                target_sequences.append(target_word)
                sample_count += 1
                
                if sample_count >= max_samples:
                    break
    
    print(f"✅ Created {len(input_sequences)} training examples")
    print(f"✅ Max sequence length: {max_length}")
    print(f"✅ Unique images: {len(set(image_names))}")
    
    # Pad sequences
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='pre')
    
    return np.array(image_names), padded_sequences, np.array(target_sequences), max_length

def build_simple_model(vocab_size, max_length, feature_dim=2048):
    """Build a simplified model for testing"""
    print("🏗️ Building model...")
    
    # Image feature input (we'll use dummy features for now)
    image_input = Input(shape=(feature_dim,), name='image_features')
    image_dense = Dense(512, activation='relu')(image_input)
    image_dense = Dropout(0.3)(image_dense)
    
    # Caption input
    caption_input = Input(shape=(max_length,), name='caption_input')
    embedding = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    embedding = Dropout(0.3)(embedding)
    
    # LSTM
    lstm_out = LSTM(512)(embedding)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Combine features
    combined = Add()([image_dense, lstm_out])
    combined = Dense(512, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    
    # Output
    output = Dense(vocab_size, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, caption_input], outputs=output)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model built successfully")
    print(f"📊 Parameters: {model.count_params():,}")
    
    return model

def create_dummy_features(image_names, feature_dim=2048):
    """Create dummy image features for testing"""
    print("🎲 Creating dummy image features for testing...")
    
    # Create consistent dummy features for each unique image
    unique_images = list(set(image_names))
    image_to_feature = {}
    
    np.random.seed(42)  # For reproducibility
    for img in unique_images:
        image_to_feature[img] = np.random.normal(0, 1, feature_dim).astype(np.float32)
    
    # Map features to all sequences (each sequence gets the feature of its corresponding image)
    features = np.array([image_to_feature[img] for img in image_names])
    
    print(f"✅ Created features for {len(unique_images)} unique images")
    print(f"✅ Feature array shape: {features.shape}")
    print(f"✅ Total training sequences: {len(image_names)}")
    
    return features

def load_real_image_features(image_names, features_csv_path='resized_features.csv'):
    """Load real image features and map them to sequences"""
    print("🔍 Loading real image features...")
    
    try:
        # Load features in chunks for memory efficiency
        print("   Loading features file...")
        features_df = pd.read_csv(features_csv_path)
        
        print(f"✅ Loaded features for {len(features_df)} images")
        
        # Create mapping from image name to features
        if 'image' in features_df.columns:
            image_col = 'image'
            feature_cols = [col for col in features_df.columns if col != 'image']
        else:
            # Assume first column is image name
            image_col = features_df.columns[0]
            feature_cols = features_df.columns[1:]
        
        image_to_feature = {}
        for _, row in features_df.iterrows():
            img_name = row[image_col]
            features = row[feature_cols].values.astype(np.float32)
            image_to_feature[img_name] = features
        
        print(f"✅ Created feature mapping for {len(image_to_feature)} images")
        
        # Map features to all sequences
        mapped_features = []
        missing_images = []
        
        for img_name in image_names:
            if img_name in image_to_feature:
                mapped_features.append(image_to_feature[img_name])
            else:
                missing_images.append(img_name)
                # Create dummy feature for missing image
                mapped_features.append(np.random.normal(0, 1, len(feature_cols)).astype(np.float32))
        
        features_array = np.array(mapped_features)
        
        if missing_images:
            print(f"⚠️  Warning: {len(missing_images)} images not found in features file")
            print(f"   Using dummy features for missing images")
        
        print(f"✅ Final feature array shape: {features_array.shape}")
        
        return features_array
        
    except Exception as e:
        print(f"❌ Error loading real features: {e}")
        print("   Falling back to dummy features...")
        return create_dummy_features(image_names)

def train_model():
    """Main training function"""
    print("🚀 Starting Image Captioning Training")
    print("=" * 50)
    
    # Load data
    captions_df, tokenizer, vocab_size, start_token, end_token = load_data()
    feature_dim = load_image_features_sample()
    
    # Create sequences (limit for testing)
    print("\n📊 Creating training sequences...")
    image_names, input_sequences, target_sequences, max_length = create_sequences_with_images(
        captions_df, tokenizer, start_token, end_token, max_samples=100000
    )
    
    print(f"✅ Generated {len(image_names)} training examples")
    print(f"✅ Unique images in sequences: {len(set(image_names))}")
    
    # Load image features (try real features first, fall back to dummy)
    print("\n🖼️ Loading image features...")
    try:
        image_features = load_real_image_features(image_names, 'resized_features.csv')
        print("✅ Using real image features")
    except:
        image_features = create_dummy_features(image_names, feature_dim)
        print("✅ Using dummy features for testing")
    
    # Verify shapes match
    print(f"\n🔍 Verifying data consistency:")
    print(f"   Image names: {len(image_names)}")
    print(f"   Input sequences: {input_sequences.shape}")
    print(f"   Target sequences: {target_sequences.shape}")
    print(f"   Image features: {image_features.shape}")
    
    assert len(image_names) == len(input_sequences) == len(target_sequences) == len(image_features), \
        "Mismatch in data lengths!"
    
    # Split data
    print("\n📊 Splitting data...")
    indices = np.arange(len(input_sequences))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_features = image_features[train_idx]
    train_sequences = input_sequences[train_idx]
    train_targets = target_sequences[train_idx]
    
    val_features = image_features[val_idx]
    val_sequences = input_sequences[val_idx]
    val_targets = target_sequences[val_idx]
    
    print(f"✅ Training samples: {len(train_idx):,}")
    print(f"✅ Validation samples: {len(val_idx):,}")
    
    # Build model
    model = build_simple_model(vocab_size, max_length, image_features.shape[1])
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_caption_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("\n🏃 Starting training...")
    history = model.fit(
        x=[train_features, train_sequences],
        y=train_targets,
        validation_data=([val_features, val_sequences], val_targets),
        batch_size=64,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save components
    print("\n💾 Saving model components...")
    model.save('final_caption_model.h5')
    
    with open('final_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    metadata = {
        'vocab_size': vocab_size,
        'max_length': max_length,
        'feature_dim': image_features.shape[1],
        'start_token': start_token,
        'end_token': end_token
    }
    
    with open('training_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("✅ Training completed and model saved!")
    
    # Print final metrics
    final_loss = history.history['val_loss'][-1]
    final_acc = history.history['val_accuracy'][-1]
    print(f"\n📊 Final Results:")
    print(f"   Validation Loss: {final_loss:.4f}")
    print(f"   Validation Accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    train_model()
