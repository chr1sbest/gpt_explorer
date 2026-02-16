# Contributing to GPT Token Explorer

Thank you for your interest in contributing! This is an educational project aimed at making GPT token generation accessible and fun to explore.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/gpt-token-explorer.git`
3. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt && pip install -r requirements-dev.txt`
5. Create a feature branch: `git checkout -b feature/amazing-feature`

## Development

### Running Tests

```bash
source venv/bin/activate
# On Apple Silicon (ARM64), use:
arch -arm64 python -m pytest tests/ -v
# On other platforms:
python -m pytest tests/ -v
```

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable and function names
- Add docstrings to all classes and methods
- Keep functions focused and concise

### Testing Your Changes

Before submitting a PR:

1. Run all tests: `arch -arm64 python -m pytest tests/` (or `python -m pytest tests/` on non-ARM platforms)
2. Test the REPL manually with `./explore.sh`
3. Verify the README examples still work
4. Check that error handling works correctly

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation/docstrings as needed
5. Create a PR with a clear title and description

## Ideas for Contributions

- Add new commands (e.g., compare models side-by-side, export results)
- Improve error messages and user experience
- Add visualization features (probability heatmaps, attention visualization)
- Support for more pre-trained models (GPT-2, other LLMs)
- Add sampling strategies (temperature, top-k, nucleus sampling)
- Performance optimizations
- Better test coverage
- Documentation improvements
- Bug fixes

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Remember this is an educational project - clarity over cleverness

## Questions?

Feel free to open an issue for any questions or suggestions!
