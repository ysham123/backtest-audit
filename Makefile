PYTHON ?= python3

.PHONY: all test experiments report clean

all: test experiments

test:
	$(PYTHON) -m pytest -q

experiments:
	$(PYTHON) -m experiments.run_all

report:
	$(PYTHON) -m experiments.report --write

clean:
	rm -rf .pytest_cache build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
