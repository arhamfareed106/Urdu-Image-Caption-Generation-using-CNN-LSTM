# Usage Guide

## Quick Start

### 1. Basic Usage

```python
from urdu_caption import UrduCaptionGenerator

# Initialize the model
generator = UrduCaptionGenerator()

# Generate caption for an image
image_path = "path/to/your/image.jpg"
caption = generator.generate_caption(image_path)
print(f"Generated Caption: {caption}")
```

### 2. Display Image with Caption

```python
import matplotlib.pyplot as plt
from PIL import Image

# Load and display image with caption
img = Image.open(image_path)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title(caption, fontsize=14, pad=20)
plt.axis('off')
plt.show()
```

## Advanced Usage

### Batch Processing

```python
# Process multiple images
image_paths = [
    "image1.jpg",
    "image2.jpg", 
    "image3.jpg"
]

captions = generator.generate_captions_batch(image_paths)

for image_path, caption in zip(image_paths, captions):
    print(f"{image_path}: {caption}")
```

### Custom Parameters

```python
# Generate caption with custom parameters
caption = generator.generate_caption(
    image_path=image_path,
    max_length=50,          # Maximum caption length
    temperature=0.8,        # Sampling temperature (0.1-1.0)
    top_k=5,               # Top-k sampling
    beam_size=3            # Beam search size
)
```

## Model Configuration

### Loading Pre-trained Model

```python
from urdu_caption import UrduCaptionGenerator

# Load specific model
generator = UrduCaptionGenerator(
    model_path="models/urdu_caption_model.keras",
    tokenizer_path="models/urdu_tokenizer.pkl",
    config_path="config/model_config.json"
)
```

### Custom Model Configuration

```python
config = {
    "vocab_size": 11077,
    "max_length": 59,
    "embedding_dim": 256,
    "lstm_units": 256,
    "dropout_rate": 0.5,
    "beam_size": 3,
    "temperature": 0.8
}

generator = UrduCaptionGenerator(config=config)
```

## Text Processing

### Urdu Text Rendering

```python
from utils.text_utils import reshape_urdu_text

# Proper Urdu text display
urdu_caption = "ایک کمرہ جس میں نیلی دیواروں اور سفید سنک ہے"
display_text = reshape_urdu_text(urdu_caption)

plt.title(display_text, fontsize=12, fontfamily='Arial')
```

### Text Cleaning

```python
from utils.text_utils import clean_urdu_text

# Clean Urdu text
raw_text = "ایک کمرہ جس میں نیلی دیواروں اور سفید سنک ہے۔"
clean_text = clean_urdu_text(raw_text)
```

## Performance Optimization

### GPU Acceleration

```python
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
# Use mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### Batch Processing

```python
# Process images in batches for better performance
batch_size = 32
image_paths = ["image1.jpg", "image2.jpg", ...]

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch_captions = generator.generate_captions_batch(batch_paths)
```

## Evaluation

### Model Performance

```python
from evaluation import evaluate_model

# Evaluate model on test set
test_images = ["test1.jpg", "test2.jpg", ...]
test_captions = ["caption1", "caption2", ...]

metrics = evaluate_model(
    generator, 
    test_images, 
    test_captions
)

print(f"BLEU Score: {metrics['bleu']:.4f}")
print(f"METEOR Score: {metrics['meteor']:.4f}")
print(f"ROUGE Score: {metrics['rouge']:.4f}")
```

### Caption Quality Analysis

```python
from evaluation import analyze_captions

# Analyze generated captions
analysis = analyze_captions(generated_captions)

print(f"Average Length: {analysis['avg_length']:.2f}")
print(f"Vocabulary Diversity: {analysis['vocab_diversity']:.4f}")
print(f"Repetition Rate: {analysis['repetition_rate']:.4f}")
```

## Integration Examples

### Web Application

```python
from flask import Flask, request, jsonify
from urdu_caption import UrduCaptionGenerator

app = Flask(__name__)
generator = UrduCaptionGenerator()

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_path = save_uploaded_file(image_file)
    
    try:
        caption = generator.generate_caption(image_path)
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Desktop Application

```python
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from urdu_caption import UrduCaptionGenerator

class CaptionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.generator = UrduCaptionGenerator()
        self.setup_ui()
    
    def setup_ui(self):
        # Create UI components
        self.select_btn = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_btn.pack()
        
        self.caption_label = tk.Label(self.root, text="", wraplength=400)
        self.caption_label.pack()
    
    def select_image(self):
        image_path = filedialog.askopenfilename()
        if image_path:
            caption = self.generator.generate_caption(image_path)
            self.caption_label.config(text=caption)
    
    def run(self):
        self.root.mainloop()

app = CaptionApp()
app.run()
```

## Error Handling

### Common Errors and Solutions

```python
try:
    caption = generator.generate_caption(image_path)
except FileNotFoundError:
    print("Image file not found. Please check the path.")
except ValueError as e:
    print(f"Invalid image format: {e}")
except RuntimeError as e:
    print(f"Model loading error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Model Loading Issues

```python
# Check if model files exist
import os

model_path = "models/urdu_caption_model.keras"
tokenizer_path = "models/urdu_tokenizer.pkl"

if not os.path.exists(model_path):
    print("Model file not found. Please download or train the model.")
    
if not os.path.exists(tokenizer_path):
    print("Tokenizer file not found. Please recreate the tokenizer.")
```

## Best Practices

### 1. Image Preprocessing

```python
# Ensure images are properly formatted
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  # Convert to RGB
    img = img.resize((299, 299))  # Resize for InceptionV3
    return img
```

### 2. Memory Management

```python
# Clear GPU memory after processing
import tensorflow as tf

def clear_gpu_memory():
    tf.keras.backend.clear_session()
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices('GPU')[0], True
        )
```

### 3. Caching

```python
# Cache extracted features for better performance
import pickle

def cache_features(image_paths, cache_file="features_cache.pkl"):
    features = {}
    for path in tqdm(image_paths):
        features[path] = extract_features(path)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)
    
    return features
```

## Troubleshooting

### Performance Issues

1. **Slow Generation**: Use GPU acceleration and batch processing
2. **Memory Issues**: Reduce batch size or use memory-efficient models
3. **Poor Quality**: Adjust temperature and top-k parameters

### Text Display Issues

1. **Urdu Text Not Rendering**: Install proper Urdu fonts
2. **Right-to-Left Issues**: Use arabic-reshaper and python-bidi
3. **Font Problems**: Use system fonts or download Urdu fonts

### Model Issues

1. **Loading Errors**: Check model file integrity
2. **Vocabulary Mismatch**: Ensure consistent tokenizer
3. **Dimension Errors**: Verify input image size and format

## Support

For additional help:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review [GitHub Issues](https://github.com/yourusername/urdu-caption-generation/issues)
3. Create a new issue with detailed error information
4. Contact the development team 