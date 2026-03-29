.PHONY: install test benchmark ablation baselines paper predictions clean

install:
	pip install -e ".[dev,v2,science,benchmark]"

test:
	python -m pytest tests/ -v --tb=short

benchmark:
	python -m src.benchmark.v2_benchmark

ablation:
	python -m src.benchmark.v2_ablation

baselines:
	python -m src.benchmark.baselines

predictions:
	python -m src.benchmark.verify_scorecard

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

# One-command reproducibility
reproduce: install test benchmark ablation baselines
	@echo "All checks passed. Results in data/v2_results/"
