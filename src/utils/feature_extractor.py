"""
Feature extraction utilities for image captioning
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from PIL import Image
import pickle
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class FeatureExtractor:
    """
    Extract features from images using pre-trained CNN models
    """
    
    def __init__(self, model_name: str = "inception_v3"):
        """
        Initialize feature extractor
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.model = None
        self.feature_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained CNN model"""
        try:
            if self.model_name == "inception_v3":
                base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
                self.model = base_model
                self.feature_dim = 2048
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            print(f"✅ {self.model_name} model loaded successfully!")
            print(f"✅ Feature dimension: {self.feature_dim}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector
        """
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(299, 299))
            img_array = img_to_array(img)
            img_tensor = np.expand_dims(img_array, axis=0)
            img_tensor = preprocess_input(img_tensor)
            
            # Extract features
            features = self.model.predict(img_tensor, verbose=0)
            return features[0]  # Remove batch dimension
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            raise
    
    def extract_features_batch(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Extract features from multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping image paths to feature vectors
        """
        features_dict = {}
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load batch images
            for path in batch_paths:
                try:
                    img = load_img(path, target_size=(299, 299))
                    img_array = img_to_array(img)
                    batch_images.append(img_array)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"⚠️ Skipping {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Preprocess batch
            batch_tensor = np.array(batch_images)
            batch_tensor = preprocess_input(batch_tensor)
            
            # Extract features
            batch_features = self.model.predict(batch_tensor, verbose=0)
            
            # Store features
            for path, feature in zip(valid_paths, batch_features):
                features_dict[path] = feature
        
        return features_dict
    
    def save_features(self, features_dict: Dict[str, np.ndarray], save_path: str):
        """
        Save extracted features to file
        
        Args:
            features_dict: Dictionary of features
            save_path: Path to save features
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(features_dict, f)
            print(f"✅ Features saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving features: {e}")
            raise
    
    def load_features(self, load_path: str) -> Dict[str, np.ndarray]:
        """
        Load features from file
        
        Args:
            load_path: Path to load features from
            
        Returns:
            Dictionary of features
        """
        try:
            with open(load_path, 'rb') as f:
                features_dict = pickle.load(f)
            print(f"✅ Features loaded from {load_path}")
            return features_dict
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            raise


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (299, 299)) -> np.ndarray:
    """
    Preprocess image for feature extraction
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image array
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to array
        img_array = np.array(img)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Error preprocessing image {image_path}: {e}")
        raise


def validate_image(image_path: str) -> bool:
    """
    Validate if image file is valid and readable
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if image is valid
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return False
        
        # Try to open image
        with Image.open(image_path) as img:
            img.verify()
        
        return True
        
    except Exception:
        return False


def get_image_info(image_path: str) -> Dict:
    """
    Get information about an image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            info = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'file_size': os.path.getsize(image_path)
            }
        return info
    except Exception as e:
        print(f"❌ Error getting image info for {image_path}: {e}")
        return {}


def extract_features_from_directory(directory_path: str, 
                                  output_path: str,
                                  batch_size: int = 32) -> Dict[str, np.ndarray]:
    """
    Extract features from all images in a directory
    
    Args:
        directory_path: Path to directory containing images
        output_path: Path to save extracted features
        batch_size: Batch size for processing
        
    Returns:
        Dictionary of features
    """
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(directory_path, filename)
            if validate_image(image_path):
                image_paths.append(image_path)
    
    print(f"Found {len(image_paths)} valid images")
    
    # Extract features
    extractor = FeatureExtractor()
    features_dict = extractor.extract_features_batch(image_paths, batch_size)
    
    # Save features
    extractor.save_features(features_dict, output_path)
    
    return features_dict


def create_feature_cache(image_paths: List[str], 
                        cache_path: str,
                        batch_size: int = 32) -> Dict[str, np.ndarray]:
    """
    Create a cache of extracted features
    
    Args:
        image_paths: List of image paths
        cache_path: Path to save cache
        batch_size: Batch size for processing
        
    Returns:
        Dictionary of features
    """
    # Check if cache exists
    if os.path.exists(cache_path):
        try:
            extractor = FeatureExtractor()
            features_dict = extractor.load_features(cache_path)
            print(f"✅ Loaded {len(features_dict)} features from cache")
            return features_dict
        except Exception as e:
            print(f"⚠️ Cache corrupted, recreating: {e}")
    
    # Extract features
    extractor = FeatureExtractor()
    features_dict = extractor.extract_features_batch(image_paths, batch_size)
    
    # Save cache
    extractor.save_features(features_dict, cache_path)
    
    return features_dict


# Example usage
if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()
    
    # Test with a sample image
    sample_image = "sample_image.jpg"
    if os.path.exists(sample_image):
        features = extractor.extract_features(sample_image)
        print(f"Extracted features shape: {features.shape}")
        print(f"Feature vector: {features[:10]}...")  # Show first 10 values
    else:
        print("Sample image not found. Please provide a valid image path.") 