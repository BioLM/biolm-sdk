.PHONY: clean clean-build clean-pyc clean-test coverage dist docs docs-iframe help install lint lint/flake8
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

RS ?= 12345

k ?= 
x ?= 8

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-ruff ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -rf {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-ruff: ## remove ruff artifacts
	rm -fr .ruff_cache/

lint/ruff: ## run ruff to check Python code style
	ruff biolm/ tests/

lint/black: ## run black to check and format Python code style
	black --check biolmai/ tests/

lint/flake8: ## check style with flake8
	flake8 biolmai/ tests/

lint: lint/ruff lint/black lint/flake8 ## check style with ruff, black, flake8

setup-tox:
	pyenv install -s 3.13.3
	pyenv install -s 3.12.10
	pyenv install -s 3.11.6
	pyenv install -s 3.10.13
	pyenv install -s 3.9.22
	pyenv install -s 3.8.20
	pyenv install -s 3.7.17
	pyenv local 3.11.6 3.7.17 3.8.20 3.9.22 3.10.13 3.12.10 3.13.3


test: ## run tests quickly with the default Python
	pytest -s --durations=5 --randomly-seed="$(RS)" $(if $(k),-k "$(k)") -n 0

ptest: ## run tests quickly with the default Python
	pytest -s --durations=5 --randomly-seed="$(RS)" $(if $(k),-k "$(k)") -n $(x)

test-all: ## run tests on every Python version with tox
	tox

test-parallel: ## run tests on every Python version with tox
	tox --parallel 8

coverage: ## check code coverage quickly with the default Python
	coverage run --source biolm -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	mkdir -p docs/_static docs/api-reference
	rm -f docs/api-reference/modules.rst docs/api-reference/biolmai.rst docs/api-reference/biolmai.*.rst
	sphinx-apidoc -o docs/api-reference biolm
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

docs-iframe: ## generate docs in iframe mode (no header, for embedding)
	mkdir -p docs/_static docs/api-reference
	rm -f docs/api-reference/modules.rst docs/api-reference/biolmai.rst docs/api-reference/biolmai.*.rst
	sphinx-apidoc -o docs/api-reference biolm
	$(MAKE) -C docs clean
	IFRAME_MODE=1 $(MAKE) -C docs html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

testrelease: dist ## package and upload a release
	twine upload --repository testpypi --verbose dist/*

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	pip install -e .
