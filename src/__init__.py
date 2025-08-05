"""
Urdu Caption Generation Package
"""

from .urdu_caption import (
    UrduCaptionGenerator,
    reshape_urdu_text,
    clean_urdu_text
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'UrduCaptionGenerator',
    'reshape_urdu_text',
    'clean_urdu_text'
] 