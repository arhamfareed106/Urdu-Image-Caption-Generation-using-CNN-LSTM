"""
Utility functions for Urdu text processing
"""

import arabic_reshaper
from bidi.algorithm import get_display
import re
from typing import List, Dict, Optional


def reshape_urdu_text(text: str) -> str:
    """
    Reshape Urdu text for proper right-to-left display
    
    Args:
        text: Urdu text to reshape
        
    Returns:
        Reshaped text for proper display
    """
    try:
        if not text or not isinstance(text, str):
            return text
        
        # Reshape Arabic/Urdu characters
        reshaped = arabic_reshaper.reshape(text)
        
        # Apply bidirectional algorithm for RTL text
        bidi_text = get_display(reshaped)
        
        return bidi_text
    except Exception as e:
        print(f"Warning: Error reshaping Urdu text: {e}")
        return text


def clean_urdu_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Clean Urdu text by removing punctuation and normalizing
    
    Args:
        text: Raw Urdu text
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return text
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    if remove_punctuation:
        # Urdu and English punctuation marks
        punctuation = [
            '۔', '?', '!', ',', '،', '.', ';', ':', 
            '"', "'", '(', ')', '[', ']', '{', '}',
            '،', '؛', '؟', '۔', '،', '۔', '،', '۔'
        ]
        
        for punct in punctuation:
            text = text.replace(punct, '')
    
    return text.strip()


def add_start_end_tokens(text: str) -> str:
    """
    Add start and end tokens to text for training
    
    Args:
        text: Input text
        
    Returns:
        Text with start and end tokens
    """
    return f"startseq {text} endseq"


def remove_start_end_tokens(text: str) -> str:
    """
    Remove start and end tokens from text
    
    Args:
        text: Text with tokens
        
    Returns:
        Text without tokens
    """
    return text.replace('startseq ', '').replace(' endseq', '')


def split_urdu_sentences(text: str) -> List[str]:
    """
    Split Urdu text into sentences
    
    Args:
        text: Urdu text
        
    Returns:
        List of sentences
    """
    # Urdu sentence endings
    sentence_endings = ['۔', '!', '?', '؟']
    
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in sentence_endings:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add remaining text if any
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return [s for s in sentences if s]


def count_urdu_words(text: str) -> int:
    """
    Count words in Urdu text
    
    Args:
        text: Urdu text
        
    Returns:
        Number of words
    """
    if not text:
        return 0
    
    # Split by whitespace and filter empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)


def get_urdu_vocabulary(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    """
    Build vocabulary from Urdu texts
    
    Args:
        texts: List of Urdu texts
        min_freq: Minimum frequency for word inclusion
        
    Returns:
        Dictionary of word frequencies
    """
    word_freq = {}
    
    for text in texts:
        if not text:
            continue
        
        # Clean and split text
        clean_text = clean_urdu_text(text)
        words = clean_text.split()
        
        for word in words:
            word = word.strip()
            if word:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Filter by minimum frequency
    filtered_vocab = {word: freq for word, freq in word_freq.items() 
                     if freq >= min_freq}
    
    return filtered_vocab


def validate_urdu_text(text: str) -> bool:
    """
    Validate if text contains Urdu characters
    
    Args:
        text: Text to validate
        
    Returns:
        True if text contains Urdu characters
    """
    if not text:
        return False
    
    # Urdu Unicode range
    urdu_range = range(0x0600, 0x06FF)  # Arabic block
    urdu_supplement = range(0x0750, 0x077F)  # Arabic Supplement
    
    for char in text:
        if ord(char) in urdu_range or ord(char) in urdu_supplement:
            return True
    
    return False


def normalize_urdu_text(text: str) -> str:
    """
    Normalize Urdu text by standardizing characters
    
    Args:
        text: Urdu text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return text
    
    # Character mappings for normalization
    char_mappings = {
        'ی': 'ي',  # Standardize different forms of ye
        'ے': 'ي',
        'ہ': 'ه',  # Standardize different forms of he
        'ۂ': 'ه',
        'ؤ': 'و',  # Standardize waw
        'ئ': 'ي',  # Standardize hamza
    }
    
    normalized = text
    for old_char, new_char in char_mappings.items():
        normalized = normalized.replace(old_char, new_char)
    
    return normalized


def extract_urdu_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract most common Urdu words from text
    
    Args:
        text: Urdu text
        top_n: Number of top keywords to return
        
    Returns:
        List of top keywords
    """
    if not text:
        return []
    
    # Clean and split text
    clean_text = clean_urdu_text(text)
    words = clean_text.split()
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        word = word.strip()
        if word and len(word) > 1:  # Filter out single characters
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]


def format_urdu_caption(caption: str, max_length: int = 100) -> str:
    """
    Format Urdu caption for display
    
    Args:
        caption: Raw caption
        max_length: Maximum display length
        
    Returns:
        Formatted caption
    """
    if not caption:
        return ""
    
    # Clean caption
    clean_caption = clean_urdu_text(caption, remove_punctuation=False)
    
    # Truncate if too long
    if len(clean_caption) > max_length:
        clean_caption = clean_caption[:max_length] + "..."
    
    # Reshape for display
    return reshape_urdu_text(clean_caption)


def create_urdu_summary(text: str, max_sentences: int = 3) -> str:
    """
    Create a summary of Urdu text
    
    Args:
        text: Urdu text
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Summary text
    """
    if not text:
        return ""
    
    # Split into sentences
    sentences = split_urdu_sentences(text)
    
    # Take first few sentences
    summary_sentences = sentences[:max_sentences]
    
    # Join sentences
    summary = " ".join(summary_sentences)
    
    return format_urdu_caption(summary)


# Example usage and testing
if __name__ == "__main__":
    # Test Urdu text processing
    test_text = "ایک کمرہ جس میں نیلی دیواروں اور سفید سنک ہے۔"
    
    print("Original text:", test_text)
    print("Cleaned text:", clean_urdu_text(test_text))
    print("Reshaped text:", reshape_urdu_text(test_text))
    print("Word count:", count_urdu_words(test_text))
    print("Is Urdu text:", validate_urdu_text(test_text))
    print("Keywords:", extract_urdu_keywords(test_text, top_n=5)) 