#!/usr/bin/env python3
"""
Image Captioning Model Training Script
Handles 39,874 images with 5 captions each using encoder-decoder architecture
"""

import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True) if tf.config.experimental.list_physical_devices('GPU') else None

class ImageCaptioningTrainer:
    def __init__(self, 
                 features_path='resized_features.csv',
                 captions_path='cap_prep/cleanedcaptions1.csv',
                 tokenizer_path='cap_prep/tokenizer1.pkl'):
        
        self.features_path = features_path
        self.captions_path = captions_path
        self.tokenizer_path = tokenizer_path
        
        # Model hyperparameters
        self.embedding_dim = 256
        self.lstm_units = 512
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        
        # Training parameters
        self.batch_size = 64
        self.epochs = 50
        self.validation_split = 0.2
        
    def load_and_prepare_data(self):
        """Load and prepare all training data"""
        print("🔍 Loading and preparing data...")
        
        # Load tokenizer
        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Load captions
        captions_df = pd.read_csv(self.captions_path)
        print(f"✅ Loaded {len(captions_df)} caption records")
        
        # Prepare image-caption mapping
        image_to_captions = {}
        for _, row in captions_df.iterrows():
            image_name = row['image']
            caption = row['caption']
            
            if image_name not in image_to_captions:
                image_to_captions[image_name] = []
            image_to_captions[image_name].append(caption)
        
        print(f"✅ Found {len(image_to_captions)} unique images")
        print(f"✅ Average captions per image: {np.mean([len(caps) for caps in image_to_captions.values()]):.1f}")
        
        self.image_to_captions = image_to_captions
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        # Add special tokens if not present
        if '<start>' not in self.tokenizer.word_index:
            self.tokenizer.word_index['<start>'] = len(self.tokenizer.word_index) + 1
        if '<end>' not in self.tokenizer.word_index:
            self.tokenizer.word_index['<end>'] = len(self.tokenizer.word_index) + 1
            
        self.start_token = self.tokenizer.word_index['<start>']
        self.end_token = self.tokenizer.word_index['<end>']
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        print(f"✅ Vocabulary size: {self.vocab_size}")
        print(f"✅ Start token: {self.start_token}, End token: {self.end_token}")
        
    def load_image_features(self):
        """Load image features efficiently"""
        print("🔍 Loading image features...")
        
        # Try to load features in chunks to handle large files
        try:
            # Load features in chunks
            feature_chunks = []
            chunk_size = 10000
            
            for chunk in pd.read_csv(self.features_path, chunksize=chunk_size):
                feature_chunks.append(chunk)
            
            features_df = pd.concat(feature_chunks, ignore_index=True)
            print(f"✅ Loaded image features: {features_df.shape}")
            
            # Extract image names and feature vectors
            if 'image' in features_df.columns:
                image_names = features_df['image'].values
                feature_columns = [col for col in features_df.columns if col != 'image']
                features = features_df[feature_columns].values
            else:
                # Assume first column is image name, rest are features
                image_names = features_df.iloc[:, 0].values
                features = features_df.iloc[:, 1:].values
            
            # Create image-to-features mapping
            self.image_to_features = {}
            for img_name, feature_vec in zip(image_names, features):
                self.image_to_features[img_name] = feature_vec.astype(np.float32)
            
            self.feature_dim = features.shape[1]
            print(f"✅ Feature dimension: {self.feature_dim}")
            print(f"✅ Created feature mapping for {len(self.image_to_features)} images")
            
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            raise
    
    def create_training_sequences(self):
        """Create training sequences from captions"""
        print("🔧 Creating training sequences...")
        
        image_features = []
        input_sequences = []
        target_sequences = []
        
        max_caption_length = 0
        
        for image_name, captions in self.image_to_captions.items():
            if image_name not in self.image_to_features:
                continue
                
            image_feature = self.image_to_features[image_name]
            
            for caption in captions:
                # Tokenize caption
                caption_tokens = self.tokenizer.texts_to_sequences([caption])[0]
                
                if len(caption_tokens) == 0:
                    continue
                
                # Add start and end tokens
                full_sequence = [self.start_token] + caption_tokens + [self.end_token]
                max_caption_length = max(max_caption_length, len(full_sequence))
                
                # Create input-output pairs for each position
                for i in range(1, len(full_sequence)):
                    input_seq = full_sequence[:i]
                    target_word = full_sequence[i]
                    
                    image_features.append(image_feature)
                    input_sequences.append(input_seq)
                    target_sequences.append(target_word)
        
        print(f"✅ Created {len(input_sequences)} training examples")
        print(f"✅ Max caption length: {max_caption_length}")
        
        # Pad sequences
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        input_sequences = pad_sequences(input_sequences, maxlen=max_caption_length, padding='pre')
        
        self.max_caption_length = max_caption_length
        self.image_features = np.array(image_features)
        self.input_sequences = input_sequences
        self.target_sequences = np.array(target_sequences)
        
        print(f"✅ Image features shape: {self.image_features.shape}")
        print(f"✅ Input sequences shape: {self.input_sequences.shape}")
        print(f"✅ Target sequences shape: {self.target_sequences.shape}")
    
    def build_model(self):
        """Build the image captioning model"""
        print("🏗️ Building model architecture...")
        
        # Image feature input
        image_input = Input(shape=(self.feature_dim,), name='image_features')
        image_features = Dense(self.lstm_units, activation='relu')(image_input)
        image_features = Dropout(self.dropout_rate)(image_features)
        
        # Caption input
        caption_input = Input(shape=(self.max_caption_length,), name='caption_input')
        caption_embedding = Embedding(
            self.vocab_size, 
            self.embedding_dim, 
            mask_zero=True,
            input_length=self.max_caption_length
        )(caption_input)
        caption_embedding = Dropout(self.dropout_rate)(caption_embedding)
        
        # LSTM decoder
        lstm_output = LSTM(self.lstm_units, return_sequences=False)(caption_embedding)
        lstm_output = Dropout(self.dropout_rate)(lstm_output)
        
        # Combine image and text features
        combined = Add()([image_features, lstm_output])
        combined = Dense(self.lstm_units, activation='relu')(combined)
        combined = Dropout(self.dropout_rate)(combined)
        
        # Output layer
        output = Dense(self.vocab_size, activation='softmax', name='word_output')(combined)
        
        # Create model
        model = Model(inputs=[image_input, caption_input], outputs=output)
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Model built successfully")
        print(f"📊 Total parameters: {model.count_params():,}")
        
    def prepare_training_data(self):
        """Prepare data for training"""
        print("📊 Preparing training data...")
        
        # Split data
        indices = np.arange(len(self.input_sequences))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=self.validation_split, 
            random_state=42
        )
        
        self.train_image_features = self.image_features[train_idx]
        self.train_input_sequences = self.input_sequences[train_idx]
        self.train_targets = self.target_sequences[train_idx]
        
        self.val_image_features = self.image_features[val_idx]
        self.val_input_sequences = self.input_sequences[val_idx]
        self.val_targets = self.target_sequences[val_idx]
        
        print(f"✅ Training samples: {len(train_idx):,}")
        print(f"✅ Validation samples: {len(val_idx):,}")
    
    def train_model(self):
        """Train the model"""
        print("🚀 Starting training...")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Training
        history = self.model.fit(
            x=[self.train_image_features, self.train_input_sequences],
            y=self.train_targets,
            validation_data=([self.val_image_features, self.val_input_sequences], self.val_targets),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        print("✅ Training completed!")
        
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def save_model_components(self):
        """Save model and training components"""
        print("💾 Saving model components...")
        
        # Save model
        self.model.save('image_captioning_model.h5')
        
        # Save tokenizer
        with open('final_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save metadata
        metadata = {
            'vocab_size': self.vocab_size,
            'max_caption_length': self.max_caption_length,
            'feature_dim': self.feature_dim,
            'start_token': self.start_token,
            'end_token': self.end_token,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units
        }
        
        with open('model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("✅ Model components saved!")
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("🎯 Starting complete training pipeline...")
        print("=" * 60)
        
        try:
            self.load_and_prepare_data()
            self.load_image_features()
            self.create_training_sequences()
            self.build_model()
            self.prepare_training_data()
            
            # Print model summary
            print("\n📋 Model Architecture:")
            self.model.summary()
            
            self.train_model()
            self.plot_training_history()
            self.save_model_components()
            
            print("\n✅ Training pipeline completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {e}")
            raise

def main():
    """Main function to run training"""
    print("🚀 Image Captioning Model Training")
    print("=" * 50)
    print("📊 Dataset: 39,874 images × 5 captions each")
    print("🎯 Architecture: CNN Encoder + LSTM Decoder")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ImageCaptioningTrainer()
    
    # Run complete training
    trainer.run_complete_training()

if __name__ == "__main__":
    main()
