.PHONY: help install install-dev test test-cov lint format type clean build publish docs pre-commit

help:
	@echo "PyImgAno Development Commands"
	@echo "=============================="
	@echo "install          Install package"
	@echo "install-dev      Install package with development dependencies"
	@echo "test             Run tests"
	@echo "test-cov         Run tests with coverage"
	@echo "lint             Run linters (flake8, ruff)"
	@echo "format           Format code with black and isort"
	@echo "type             Run type checking with mypy"
	@echo "clean            Clean build artifacts"
	@echo "build            Build distribution packages"
	@echo "publish-test     Publish to Test PyPI"
	@echo "publish          Publish to PyPI"
	@echo "docs             Build documentation"
	@echo "pre-commit       Run pre-commit hooks on all files"
	@echo "all              Run format, lint, type, and test"

install:
	pip install -e .

install-dev:
	pip install -e .[dev,diffusion,docs]
	pre-commit install

test:
	pytest tests/

test-cov:
	pytest --cov=pyimgano --cov-report=term-missing --cov-report=html tests/

lint:
	flake8 pyimgano tests
	ruff check pyimgano tests

format:
	black pyimgano tests examples
	isort pyimgano tests examples

type:
	mypy pyimgano

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache .tox
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build
	twine check dist/*

publish-test: build
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*

docs:
	cd docs && make html

pre-commit:
	pre-commit run --all-files

all: format lint type test
	@echo "All checks passed!"

# Development shortcuts
dev-setup: install-dev
	@echo "Development environment ready!"

check: format lint type
	@echo "Code quality checks passed!"

ci: check test-cov
	@echo "CI checks passed!"
