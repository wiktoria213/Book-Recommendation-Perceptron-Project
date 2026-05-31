# Book Recommendation System Using Perceptron

[![python](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![ruff](https://github.com/wnowicki/pytemp/workflows/Ruff/badge.svg)](https://github.com/wnowicki/pytemp/actions?query=branch%3Amain)
[![pytest](https://github.com/wnowicki/pytemp/workflows/Pytest/badge.svg)](https://github.com/wnowicki/pytemp/actions?query=branch%3Amain)
[![pylint](https://github.com/wnowicki/pytemp/workflows/Pylint/badge.svg)](https://github.com/wnowicki/pytemp/actions?query=branch%3Amain)
[![markdown](https://github.com/wnowicki/pytemp/workflows/Markdown%20Lint/badge.svg)](https://github.com/wnowicki/pytemp/actions?query=branch%3Amain)
[![License: GPLv3](https://img.shields.io/badge/License-MIT-blue.svg)](https://license.md/licenses/mit-license/)

This project implements a simple book recommendation system using a perceptron created from scratch in Python.

The model predicts whether a user will like a book based on user and book rating statistics.

## Dataset

[Book-Crossing Dataset](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset/data)

## How to Run the Project

### 1. Clone repository

```bash
git clone https://github.com/wiktoria213/Book-Recommendation-Perceptron-Project.git
```

### 2. Open project in Visual Studio Code

### 3. Install uv environment

```bash
pip install uv
```

### 4. Install project dependencies

```bash
uv sync
```

### 5. Run the project

```bash
uv run python app/main.py
```

## Test

```shell
uv run pytest
```

## Ruff Test

```bash
uv run ruff check .
```

Copyright (c) [2026] [Wiktoria Kozdrój]
