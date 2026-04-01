# Contributing

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/turboquant-mlx-full
cd turboquant-mlx-full
pip install -e ".[dev]"
pytest
```

## Areas

- **Correctness**: Better Lloyd-Max levels, improved Hadamard
- **Performance**: Hand-written Metal kernels (see `metal_kernels/README.md`)
- **Models**: Qwen3, Mistral, LLaMA, Gemma testing
- **Benchmarks**: Perplexity on standard corpora (WikiText, C4)

## Code style

- Python 3.9+ compatible, type hints, ruff-clean (line length 100)

## Tests

```bash
pytest                    # all
pytest -n auto            # parallel
ruff check .              # lint
```
