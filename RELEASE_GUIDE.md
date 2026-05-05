# Enterprise-Grade PyPI Release Guide for EthioBBPE
# =====================================================

This document provides comprehensive instructions for building, testing, and publishing
EthioBBPE as an enterprise-grade package to PyPI.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Modernization Summary](#modernization-summary)
3. [Development Setup](#development-setup)
4. [Testing](#testing)
5. [Building the Package](#building-the-package)
6. [Publishing to PyPI](#publishing-to-pypi)
7. [Post-Release Verification](#post-release-verification)
8. [CI/CD Integration](#cicd-integration)

---

## Prerequisites

### System Requirements

- Python 3.9 or higher (3.9, 3.10, 3.11, 3.12, 3.13 supported)
- pip >= 24.0
- setuptools >= 75.0
- wheel >= 0.44.0
- twine >= 5.0.0 (for publishing)
- build >= 1.2.0 (for building)

### Accounts Required

1. **PyPI Account**: https://pypi.org/account/register/
2. **TestPyPI Account**: https://test.pypi.org/account/register/
3. **API Tokens**: Generate tokens from account settings

### Environment Variables

```bash
export PYPI_API_TOKEN="pypi-..."
export TESTPYPI_API_TOKEN="pypi-..."
```

---

## Modernization Summary

### Dependency Updates

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| tokenizers | >=0.15.0 | >=0.20.0,<0.22 | Latest stable with performance improvements |
| huggingface-hub | >=0.20.0 | >=0.25.0,<1.0 | Enhanced caching and download features |
| pytest | >=7.0.0 | >=8.0.0 | Better async support and error reporting |
| black | >=23.0.0 | >=24.0.0 | Improved formatting rules |
| mypy | >=1.0.0 | >=1.10.0 | Stricter type checking |
| sphinx | >=7.0.0 | >=8.0.0 | Modern documentation generation |

### New Development Tools

- **ruff**: Fast Python linter (replaces flake8 for most use cases)
- **pre-commit**: Git hooks for automated code quality
- **pytest-asyncio**: Async test support
- **pytest-cov**: Enhanced coverage reporting

### Code Improvements

1. **Type Hints**: Full typing support with `from __future__ import annotations`
2. **Error Handling**: Comprehensive try-except blocks with proper exception chaining
3. **Documentation**: Enhanced docstrings with examples and type information
4. **Encoding Class**: Added `__eq__`, `to_dict()` methods for better usability
5. **Python Version**: Dropped Python 3.8, added Python 3.13 support

---

## Development Setup

### 1. Clone and Navigate

```bash
cd /workspace
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
# Install in editable mode with all dev dependencies
pip install -e ".[all]"

# Or install just the package
pip install -e .

# Or install with dev dependencies only
pip install -e ".[dev]"
```

### 4. Set Up Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run on all files (optional)
pre-commit run --all-files
```

---

## Testing

### Run All Tests

```bash
# Basic test run
pytest tests/ -v

# With coverage
pytest tests/ --cov=ethiobbpe --cov-report=term-missing --cov-report=html

# With HTML coverage report
pytest tests/ --cov=ethiobbpe --cov-report=html
# Open htmlcov/index.html in browser

# Run specific test file
pytest tests/test_tokenizer.py -v

# Run specific test function
pytest tests/test_tokenizer.py::TestEthioBBPETokenizer::test_encode_single_text -v

# Run tests excluding slow tests
pytest tests/ -v -m "not slow"

# Run integration tests only
pytest tests/ -v -m integration
```

### Type Checking

```bash
# Run mypy
mypy src/ethiobbpe/

# With strict mode
mypy src/ethiobbpe/ --strict
```

### Code Formatting

```bash
# Format with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with ruff
ruff check src/ tests/

# Auto-fix with ruff
ruff check src/ tests/ --fix
```

---

## Building the Package

### 1. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info
rm -rf src/*.egg-info
```

### 2. Build Distribution Packages

```bash
# Using build (recommended)
python -m build

# Or using setup.py directly
python setup.py sdist bdist_wheel
```

### 3. Verify Build Output

You should see:
```
dist/
├── EthioBBPE-2.0.0-py3-none-any.whl
└── EthioBBPE-2.0.0.tar.gz
```

### 4. Check Package Metadata

```bash
# Inspect wheel metadata
unzip -p dist/EthioBBPE-2.0.0-py3-none-any.whl EthioBBPE-2.0.0.dist-info/METADATA

# Or use twine
twine check dist/*
```

Expected output:
```
Checking distribution: OK
```

---

## Publishing to PyPI

### Step 1: Test on TestPyPI First

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# You'll be prompted for username/password or API token
# Username: __token__
# Password: pypi-...
```

### Step 2: Install from TestPyPI and Verify

```bash
# Create fresh virtual environment
python -m venv test-env
source test-env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    EthioBBPE==2.0.0

# Test the installation
python -c "from ethiobbpe import EthioBBPETokenizer; print('Success!')"
```

### Step 3: Publish to Production PyPI

```bash
# Upload to PyPI
twine upload dist/*

# Or specify repository explicitly
twine upload --repository pypi dist/*
```

### Step 4: Verify on PyPI

Visit: https://pypi.org/project/EthioBBPE/

Check:
- ✅ Version number is correct (2.0.0)
- ✅ Description renders properly
- ✅ Classifiers are displayed
- ✅ Download links work
- ✅ Dependencies are listed correctly

---

## Post-Release Verification

### Installation Test

```bash
# Fresh installation
pip install --upgrade EthioBBPE

# Verify version
python -c "import ethiobbpe; print(ethiobbpe.__version__)"
# Should print: 2.0.0

# Test basic functionality
python << 'EOF'
from ethiobbpe import EthioBBPETokenizer

# Load tokenizer
tokenizer = EthioBBPETokenizer.from_pretrained()

# Test encoding
text = "ሰላም ለኢዮብ"
encoded = tokenizer.encode(text)
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")

# Test decoding
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: {decoded}")
assert decoded == text, "Round-trip failed!"

print("✅ All tests passed!")
EOF
```

### Compatibility Testing

Test on multiple Python versions:

```bash
# Using pyenv or docker
docker run --rm -it python:3.9-slim bash -c "pip install EthioBBPE && python -c 'import ethiobbpe'"
docker run --rm -it python:3.10-slim bash -c "pip install EthioBBPE && python -c 'import ethiobbpe'"
docker run --rm -it python:3.11-slim bash -c "pip install EthioBBPE && python -c 'import ethiobbpe'"
docker run --rm -it python:3.12-slim bash -c "pip install EthioBBPE && python -c 'import ethiobbpe'"
docker run --rm -it python:3.13-slim bash -c "pip install EthioBBPE && python -c 'import ethiobbpe'"
```

---

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release to PyPI

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/EthioBBPE
    permissions:
      id-token: write  # IMPORTANT: for trusted publishing

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - name: Install build tools
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Check package
      run: twine check dist/*

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### Automated Testing Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Lint with ruff
      run: ruff check src/ tests/

    - name: Check formatting with black
      run: black --check src/ tests/

    - name: Type check with mypy
      run: mypy src/ethiobbpe/

    - name: Run tests
      run: pytest tests/ -v --cov=ethiobbpe --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
```

---

## Troubleshooting

### Common Issues

#### 1. Upload Authentication Failed

```bash
# Use API token instead of password
twine upload -u __token__ -p pypi-... dist/*
```

#### 2. Package Already Exists

```bash
# Bump version in pyproject.toml and __init__.py
# Then rebuild and republish
```

#### 3. Missing Files in Distribution

```toml
# Ensure pyproject.toml includes:
[tool.setuptools.package-data]
"ethiobbpe" = ["*.json", "*.json.gz", "*.txt", "*.md"]
```

#### 4. Import Errors After Installation

```bash
# Check package structure
unzip -l dist/EthioBBPE-2.0.0-py3-none-any.whl

# Ensure src/ layout is correct
```

---

## Best Practices

### Version Management

Follow Semantic Versioning (SemVer):
- **MAJOR** (2.0.0): Breaking changes
- **MINOR** (2.1.0): New features, backward compatible
- **PATCH** (2.0.1): Bug fixes, backward compatible

Update version in:
1. `pyproject.toml` → `version = "2.0.0"`
2. `src/ethiobbpe/__init__.py` → `__version__ = "2.0.0"`

### Release Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run all tests
- [ ] Check code coverage (>80%)
- [ ] Run pre-commit hooks
- [ ] Build package
- [ ] Test on TestPyPI
- [ ] Create Git tag
- [ ] Publish to PyPI
- [ ] Verify installation
- [ ] Update documentation
- [ ] Announce release

### Security Considerations

1. **API Tokens**: Never commit tokens to Git
2. **Trusted Publishing**: Use OIDC for GitHub Actions
3. **Dependencies**: Regularly audit with `pip-audit`
4. **Sign Releases**: Consider GPG signing tags

```bash
# Audit dependencies
pip install pip-audit
pip-audit

# Sign Git tag
git tag -s v2.0.0 -m "Release version 2.0.0"
```

---

## Support & Resources

- **PyPI Documentation**: https://pypi.org/help/
- **Packaging Guide**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/
- **Build Documentation**: https://pypa-build.readthedocs.io/

---

*Last Updated: 2024*
*Version: 2.0.0*
