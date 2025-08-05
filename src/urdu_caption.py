"""
Urdu Caption Generation Model
Main class for generating Urdu captions from images
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from PIL import Image
import arabic_reshaper
from bidi.algorithm import get_display
from typing import List, Dict, Optional, Union


class UrduCaptionGenerator:
    """
    Main class for Urdu image caption generation using CNN-LSTM architecture
    """
    
    def __init__(self, 
                 model_path: str = "models/urdu_caption_model.keras",
                 tokenizer_path: str = "models/urdu_tokenizer.pkl",
                 config: Optional[Dict] = None):
        """
        Initialize the Urdu Caption Generator
        
        Args:
            model_path: Path to the trained model file
            tokenizer_path: Path to the tokenizer file
            config: Configuration dictionary
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.cnn_model = None
        self.vocab_size = None
        self.max_length = None
        
        # Load model and tokenizer
        self._load_components()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "vocab_size": 11077,
            "max_length": 59,
            "embedding_dim": 256,
            "lstm_units": 256,
            "dropout_rate": 0.5,
            "beam_size": 3,
            "temperature": 0.8,
            "top_k": 5
        }
    
    def _load_components(self):
        """Load model, tokenizer, and CNN feature extractor"""
        try:
            # Load CNN feature extractor
            self._load_cnn_model()
            
            # Load tokenizer
            self._load_tokenizer()
            
            # Load caption generation model
            self._load_caption_model()
            
            print("✅ All components loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading components: {e}")
            raise
    
    def _load_cnn_model(self):
        """Load CNN model for feature extraction"""
        try:
            base_model = InceptionV3(weights='imagenet')
            self.cnn_model = Model(
                inputs=base_model.input, 
                outputs=base_model.layers[-2].output
            )
            print("✅ CNN model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading CNN model: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load or create tokenizer"""
        try:
            if os.path.exists(self.tokenizer_path):
                with open(self.tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print("✅ Tokenizer loaded from file!")
            else:
                print("⚠️ Tokenizer file not found. Creating new tokenizer...")
                self._create_tokenizer()
            
            # Update vocab size
            self.vocab_size = len(self.tokenizer.word_index) + 1
            print(f"✅ Vocabulary size: {self.vocab_size}")
            
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise
    
    def _create_tokenizer(self):
        """Create new tokenizer if file doesn't exist"""
        self.tokenizer = Tokenizer(oov_token="<unk>")
        print("✅ New tokenizer created!")
    
    def _load_caption_model(self):
        """Load the caption generation model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path, compile=False)
                print("✅ Caption model loaded successfully!")
            else:
                print("⚠️ Model file not found. Creating new model...")
                self._create_model()
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _create_model(self):
        """Create new model if file doesn't exist"""
        from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
        
        # Image input (2048-d features)
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(self.config["dropout_rate"])(inputs1)
        fe2 = Dense(self.config["embedding_dim"], activation='relu')(fe1)
        
        # Caption input (sequence)
        inputs2 = Input(shape=(self.config["max_length"],))
        se1 = Embedding(self.vocab_size, self.config["embedding_dim"], mask_zero=True)(inputs2)
        se2 = LSTM(self.config["lstm_units"])(se1)
        
        # Merge image + caption
        decoder1 = Add()([fe2, se2])
        decoder2 = Dense(self.config["embedding_dim"], activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        
        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        print("✅ New model created!")
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from image using CNN
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector (2048-dimensional)
        """
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(299, 299))
            img_array = img_to_array(img)
            img_tensor = np.expand_dims(img_array, axis=0)
            img_tensor = preprocess_input(img_tensor)
            
            # Extract features
            features = self.cnn_model.predict(img_tensor, verbose=0)
            return features[0]  # Remove batch dimension
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            raise
    
    def generate_caption(self, 
                        image_path: str, 
                        max_length: Optional[int] = None,
                        temperature: float = 0.8,
                        top_k: int = 5,
                        beam_size: int = 3) -> str:
        """
        Generate Urdu caption for an image
        
        Args:
            image_path: Path to the image file
            max_length: Maximum caption length
            temperature: Sampling temperature (0.1-1.0)
            top_k: Top-k sampling parameter
            beam_size: Beam search size
            
        Returns:
            Generated Urdu caption
        """
        try:
            # Extract image features
            features = self.extract_features(image_path)
            
            # Generate caption
            if beam_size > 1:
                caption = self._beam_search_generate(features, max_length, beam_size)
            else:
                caption = self._greedy_generate(features, max_length, temperature, top_k)
            
            return caption
            
        except Exception as e:
            print(f"❌ Error generating caption: {e}")
            raise
    
    def _greedy_generate(self, 
                        features: np.ndarray, 
                        max_length: Optional[int] = None,
                        temperature: float = 0.8,
                        top_k: int = 5) -> str:
        """Generate caption using greedy search with temperature sampling"""
        
        max_length = max_length or self.config["max_length"]
        in_text = 'startseq'
        
        for _ in range(max_length):
            # Tokenize input text
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            
            # Predict next word
            yhat = self.model.predict([np.array([features]), sequence], verbose=0)
            
            # Apply temperature and top-k sampling
            yhat = yhat[0] / temperature
            top_indices = np.argsort(yhat)[-top_k:]
            top_probs = yhat[top_indices]
            top_probs = top_probs / np.sum(top_probs)
            
            # Sample from top-k
            chosen_index = np.random.choice(top_indices, p=top_probs)
            
            # Get word
            word = self.tokenizer.index_word.get(chosen_index, '<unk>')
            
            # Stop if end token
            if word == 'endseq':
                break
                
            in_text += ' ' + word
        
        # Clean caption
        caption = in_text.replace('startseq ', '').replace(' endseq', '')
        return caption
    
    def _beam_search_generate(self, 
                             features: np.ndarray, 
                             max_length: Optional[int] = None,
                             beam_size: int = 3) -> str:
        """Generate caption using beam search"""
        
        max_length = max_length or self.config["max_length"]
        
        # Initialize beam
        beams = [('startseq', 0.0)]
        
        for _ in range(max_length):
            new_beams = []
            
            for beam_text, beam_score in beams:
                if beam_text.endswith('endseq'):
                    new_beams.append((beam_text, beam_score))
                    continue
                
                # Tokenize and predict
                sequence = self.tokenizer.texts_to_sequences([beam_text])[0]
                sequence = pad_sequences([sequence], maxlen=max_length)
                yhat = self.model.predict([np.array([features]), sequence], verbose=0)[0]
                
                # Get top beam_size words
                top_indices = np.argsort(yhat)[-beam_size:]
                
                for idx in top_indices:
                    word = self.tokenizer.index_word.get(idx, '<unk>')
                    new_text = beam_text + ' ' + word
                    new_score = beam_score + np.log(yhat[idx])
                    new_beams.append((new_text, new_score))
            
            # Keep top beam_size beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Check if all beams end with endseq
            if all(beam[0].endswith('endseq') for beam in beams):
                break
        
        # Return best caption
        best_caption = beams[0][0]
        caption = best_caption.replace('startseq ', '').replace(' endseq', '')
        return caption
    
    def generate_captions_batch(self, 
                               image_paths: List[str],
                               max_length: Optional[int] = None,
                               temperature: float = 0.8,
                               top_k: int = 5) -> List[str]:
        """
        Generate captions for multiple images
        
        Args:
            image_paths: List of image paths
            max_length: Maximum caption length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            List of generated captions
        """
        captions = []
        
        for image_path in image_paths:
            try:
                caption = self.generate_caption(
                    image_path, max_length, temperature, top_k
                )
                captions.append(caption)
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                captions.append("Error generating caption")
        
        return captions
    
    def save_model(self, model_path: Optional[str] = None):
        """Save the trained model"""
        save_path = model_path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"✅ Model saved to {save_path}")
    
    def save_tokenizer(self, tokenizer_path: Optional[str] = None):
        """Save the tokenizer"""
        save_path = tokenizer_path or self.tokenizer_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"✅ Tokenizer saved to {save_path}")
    
    def fit_tokenizer(self, captions: List[str]):
        """Fit tokenizer on new captions"""
        self.tokenizer.fit_on_texts(captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print(f"✅ Tokenizer fitted on {len(captions)} captions")
        print(f"✅ New vocabulary size: {self.vocab_size}")


def reshape_urdu_text(text: str) -> str:
    """
    Reshape Urdu text for proper display
    
    Args:
        text: Urdu text to reshape
        
    Returns:
        Reshaped text for display
    """
    try:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text
    except Exception as e:
        print(f"❌ Error reshaping Urdu text: {e}")
        return text


def clean_urdu_text(text: str) -> str:
    """
    Clean Urdu text by removing punctuation
    
    Args:
        text: Raw Urdu text
        
    Returns:
        Cleaned text
    """
    # Remove Urdu and English punctuation
    punctuation = ['۔', '?', '!', ',', '،', '.', ';', ':', '"', "'", '(', ')', '[', ']']
    for punct in punctuation:
        text = text.replace(punct, '')
    
    return text.strip()


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = UrduCaptionGenerator()
    
    # Test with sample image
    image_path = "sample_image.jpg"
    if os.path.exists(image_path):
        caption = generator.generate_caption(image_path)
        print(f"Generated Caption: {caption}")
        
        # Display with proper Urdu rendering
        display_text = reshape_urdu_text(caption)
        print(f"Display Text: {display_text}")
    else:
        print("Sample image not found. Please provide a valid image path.") 