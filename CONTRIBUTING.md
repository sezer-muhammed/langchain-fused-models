# Contributing to langchain-fused-model

Thank you for your interest in contributing to langchain-fused-model! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/langchain-fused-model.git
   cd langchain-fused-model
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in development mode with dev dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify the installation**
   ```bash
   pytest
   ```

### Project Structure

```
langchain-fused-model/
├── src/langchain_fused_model/  # Main package code
├── tests/                       # Test files
├── examples/                    # Example notebooks
├── .github/workflows/           # CI/CD workflows
└── docs/                        # Documentation (if applicable)
```

## Development Workflow

### 1. Create a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

### 2. Make Your Changes

- Write clean, readable code following the project's style guidelines
- Add or update tests for your changes
- Update documentation as needed
- Ensure all tests pass locally

### 3. Run Tests and Linting

Before submitting your changes, run the test suite and linting:

```bash
# Run tests with coverage
pytest

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: description of your changes"
```

Follow these commit message guidelines:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests when relevant

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- A clear title describing the change
- A detailed description of what changed and why
- References to related issues
- Screenshots or examples if applicable

## Coding Standards

### Style Guide

This project uses:
- **Ruff** for linting and code formatting
- **MyPy** for type checking
- **Black-compatible** formatting (via Ruff)

Key style points:
- Maximum line length: 100 characters
- Use type hints for all function signatures
- Follow PEP 8 naming conventions
- Write docstrings for all public classes and methods

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose
    and behavior of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix (e.g., `test_manager.py`)
- Use descriptive test function names that explain what is being tested
- Use pytest fixtures for common setup (see `conftest.py`)
- Aim for high test coverage, especially for core functionality

### Test Structure

```python
def test_feature_behavior():
    """Test that feature behaves correctly under normal conditions."""
    # Arrange
    setup_code()
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected_value
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_manager.py

# Run with coverage report
pytest --cov=langchain_fused_model --cov-report=html

# Run specific test
pytest tests/test_manager.py::test_specific_function
```

## Pull Request Process

1. **Ensure CI passes**: All GitHub Actions workflows must pass
2. **Update documentation**: Include relevant documentation updates
3. **Add tests**: New features must include tests
4. **Update CHANGELOG**: Add an entry describing your changes (if applicable)
5. **Request review**: Tag maintainers for review
6. **Address feedback**: Respond to review comments and make requested changes
7. **Squash commits**: Maintainers may ask you to squash commits before merging

## Reporting Issues

### Bug Reports

When reporting bugs, include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Python version and package versions
- Minimal code example demonstrating the issue
- Error messages and stack traces

### Feature Requests

When requesting features, include:
- A clear description of the feature
- Use cases and motivation
- Proposed API or interface (if applicable)
- Any alternative solutions you've considered

## Questions and Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Documentation**: Check the README and examples first

## License

By contributing to langchain-fused-model, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in the project's README and release notes.

Thank you for contributing to langchain-fused-model! 🎉
