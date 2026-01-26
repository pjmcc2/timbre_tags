# Makefile for running tests and experiments

# Default target
.PHONY: help
help:
	@echo "Available commands"
	@echo "  make test         Run all unit tests"
	@echo "  make test-int     Run integration tests"
	@echo "  make test-mock    Run mock unit tests"
	@echo "  make run-mock     Run the mock experiment"
	@echo "  make clean        Remove temporary files"

# Run all tests
.PHONY: test
test:
	PYTHONPATH=. pytest tests/ --ignore=tests/mock --ignore=tests/integration

# Run a specific test file
.PHONY: test-int
test-int:
	PYTHONPATH=. pytest tests/integration

# Run mock tests
.PHONY: test-mock
test-mock:
	PYTHONPATH=. pytest /tests/mock


# Run the mock experiment
.PHONY: run-mock
run-mock:
	python scripts/run_experiment.py --config configs/experiment/mock_experiment.yaml

# Clean up
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +