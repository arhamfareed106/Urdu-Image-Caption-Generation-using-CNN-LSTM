# Contributing to Urdu Caption Generation

Thank you for your interest in contributing to the Urdu Caption Generation project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports**: Report issues and bugs
- 💡 **Feature Requests**: Suggest new features
- 📝 **Documentation**: Improve or add documentation
- 🔧 **Code Improvements**: Fix bugs or add features
- 🧪 **Testing**: Add tests or improve test coverage
- 🌐 **Localization**: Help with Urdu language support
- 📊 **Performance**: Optimize model performance

## 🚀 Getting Started

### Prerequisites

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/urdu-caption-generation.git
   cd urdu-caption-generation
   ```
3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

### Development Setup

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
3. **Test your changes**
4. **Commit your changes**
5. **Push to your fork**
6. **Create a Pull Request**

## 📋 Development Guidelines

### Code Style

We follow PEP 8 style guidelines:

```python
# Good
def generate_caption(image_path, max_length=59):
    """Generate caption for given image.
    
    Args:
        image_path (str): Path to the image file
        max_length (int): Maximum caption length
        
    Returns:
        str: Generated Urdu caption
    """
    # Implementation here
    pass

# Bad
def generateCaption(imagePath,maxLength=59):
    # Implementation here
    pass
```

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Comments**: Add comments for complex logic
- **README**: Update README for new features
- **Type Hints**: Use type hints for function parameters

### Testing

Write tests for new features:

```python
import unittest
from urdu_caption import UrduCaptionGenerator

class TestUrduCaptionGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = UrduCaptionGenerator()
    
    def test_generate_caption(self):
        """Test caption generation for a sample image."""
        image_path = "tests/data/sample_image.jpg"
        caption = self.generator.generate_caption(image_path)
        
        self.assertIsInstance(caption, str)
        self.assertGreater(len(caption), 0)
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path."""
        with self.assertRaises(FileNotFoundError):
            self.generator.generate_caption("nonexistent.jpg")

if __name__ == '__main__':
    unittest.main()
```

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
```
feat(model): add beam search for caption generation
fix(tokenizer): resolve vocabulary size mismatch
docs(readme): update installation instructions
test(evaluation): add BLEU score calculation
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 🐛 Bug Reports

### Before Submitting

1. **Check existing issues** to avoid duplicates
2. **Test with latest version** from main branch
3. **Reproduce the issue** with minimal code

### Bug Report Template

```markdown
## Bug Description
Brief description of the issue.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python: [e.g., 3.8.10]
- TensorFlow: [e.g., 2.8.0]
- GPU: [e.g., NVIDIA RTX 3080]

## Additional Information
- Error messages
- Screenshots
- Code snippets
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Brief description of the requested feature.

## Use Case
How would this feature be used?

## Proposed Implementation
Optional: suggest how to implement.

## Alternatives Considered
Optional: other approaches you considered.

## Additional Context
Any other relevant information.
```

## 🔧 Pull Request Process

### Before Submitting PR

1. **Update documentation** for new features
2. **Add tests** for new functionality
3. **Ensure all tests pass**
4. **Update requirements** if needed
5. **Check code style** with linters

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes

## Screenshots
If applicable, add screenshots.

## Related Issues
Closes #123
```

### PR Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Address feedback** and make changes
4. **Get approval** from maintainers
5. **Merge** when ready

## 🌐 Urdu Language Support

### Text Processing Guidelines

1. **Right-to-Left Rendering**
   ```python
   import arabic_reshaper
   from bidi.algorithm import get_display
   
   def process_urdu_text(text):
       reshaped = arabic_reshaper.reshape(text)
       return get_display(reshaped)
   ```

2. **Character Encoding**
   - Use UTF-8 encoding
   - Handle Unicode normalization
   - Support Urdu punctuation

3. **Font Support**
   - Recommend Urdu fonts
   - Handle font fallbacks
   - Test rendering across platforms

### Urdu-Specific Testing

```python
def test_urdu_text_processing():
    """Test Urdu text processing functions."""
    urdu_text = "ایک کمرہ جس میں نیلی دیواروں اور سفید سنک ہے"
    
    # Test text cleaning
    cleaned = clean_urdu_text(urdu_text)
    assert "۔" not in cleaned
    
    # Test text reshaping
    reshaped = reshape_urdu_text(urdu_text)
    assert len(reshaped) > 0
```

## 📊 Performance Contributions

### Model Optimization

1. **Memory Efficiency**
   - Reduce model size
   - Optimize data loading
   - Implement caching

2. **Speed Improvements**
   - GPU optimization
   - Batch processing
   - Parallel processing

3. **Accuracy Enhancement**
   - Better architectures
   - Improved training
   - Data augmentation

### Performance Testing

```python
import time
import psutil

def benchmark_performance():
    """Benchmark model performance."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Run model inference
    caption = generator.generate_caption(image_path)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    print(f"Inference time: {end_time - start_time:.2f}s")
    print(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_utils.py
├── integration/
│   ├── test_training.py
│   └── test_inference.py
├── data/
│   └── sample_images/
└── conftest.py
```

### Test Coverage

- **Unit Tests**: Test individual functions
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test speed and memory usage

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=urdu_caption

# Run specific test file
python -m pytest tests/unit/test_model.py

# Run with verbose output
python -m pytest -v
```

## 📝 Documentation

### Documentation Standards

1. **Code Documentation**
   - Clear docstrings
   - Type hints
   - Examples

2. **User Documentation**
   - Installation guide
   - Usage examples
   - Troubleshooting

3. **Developer Documentation**
   - Architecture overview
   - API reference
   - Contributing guide

### Documentation Tools

- **Sphinx**: For API documentation
- **Jupyter Notebooks**: For tutorials
- **Markdown**: For guides and README

## 🏷️ Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. **Update version** in setup.py
2. **Update changelog**
3. **Run full test suite**
4. **Update documentation**
5. **Create release tag**
6. **Publish to PyPI**

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** to all contributors
- **Be inclusive** and welcoming
- **Be constructive** in feedback
- **Be patient** with newcomers

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions
- **Pull Requests**: For code contributions
- **Email**: For sensitive issues

## 📞 Getting Help

### Resources

- **Documentation**: Check the docs folder
- **Issues**: Search existing issues
- **Discussions**: Ask questions in discussions
- **Wiki**: Check project wiki

### Contact

- **Maintainers**: @maintainer1, @maintainer2
- **Email**: project@example.com
- **Discord**: Join our community server

## 🙏 Recognition

### Contributors

We recognize contributors in several ways:

- **Contributors list** in README
- **Release notes** for significant contributions
- **Special thanks** for major features
- **Badges** for different contribution types

### Hall of Fame

Special recognition for:

- **First-time contributors**
- **Major feature developers**
- **Long-term maintainers**
- **Community leaders**

---

Thank you for contributing to Urdu Caption Generation! 🎉 