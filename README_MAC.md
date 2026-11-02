# AI-Based Fake News Detection System — Mac M1 + VS Code

A beginner-friendly implementation of the minor project using Python, Pandas,
Scikit-learn, TF-IDF, Logistic Regression and Streamlit.

## 1. Requirements

- Mac with Apple Silicon (M1)
- VS Code
- Python 3.10–3.12 recommended
- The two dataset files: `Fake.csv` and `True.csv`

No GPU is required.

## 2. Open the project

1. Extract this ZIP file.
2. Open VS Code.
3. Select **File → Open Folder**.
4. Select `Fake_News_Detection_Mac_VSCode`.

## 3. Create a virtual environment

Open **Terminal → New Terminal** inside VS Code and run:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3` is not found, install a current Python 3 version from
https://www.python.org/downloads/macos/ and reopen VS Code.

## 4. Add your datasets

Put these exact files inside the `dataset` folder:

```text
dataset/
├── Fake.csv
└── True.csv
```

The CSV files must contain a `text` column. A `title` column is optional.
The program assigns the Fake label to rows from Fake.csv and Real to rows
from True.csv.

The dataset is not included in this project ZIP; use the dataset supplied
for your project and make sure you are permitted to use it.

## 5. Train the model

With `.venv` activated, run:

```bash
python train_model.py
```

This trains a TF-IDF + Logistic Regression pipeline and creates:

```text
models/
├── fake_news_pipeline.joblib
└── metrics.json
```

Training can take a little time depending on dataset size.

## 6. Start the website

```bash
streamlit run app.py
```

Streamlit will show a local address, usually:

`http://localhost:8501`

Open that address in your browser. To stop the app, press `Control + C`
in the VS Code terminal.

## 7. Every time you reopen VS Code

```bash
cd path/to/Fake_News_Detection_Mac_VSCode
source .venv/bin/activate
streamlit run app.py
```

You only need to run `train_model.py` again if you change the dataset or
want to retrain the model.

## Troubleshooting

**`command not found: python`**
Use `python3` to create the environment, activate `.venv`, then use
`python` inside the activated environment.

**`No module named streamlit`**
Activate the environment and run `pip install -r requirements.txt`.

**Dataset file missing**
Check spelling and capitalization: `Fake.csv` and `True.csv` must be inside
the project's `dataset` folder.

**CSV has no `text` column**
Open the CSV and check its headers. This starter project expects a `text`
column; `title` is optional.

**Model file not found**
Run `python train_model.py` successfully before starting the app.

## Project method

1. Load labelled examples from Fake.csv and True.csv.
2. Combine title (if present) and article text.
3. Remove empty and duplicate article text.
4. Split into training and testing data (80/20, stratified).
5. Fit TF-IDF and Logistic Regression as one Scikit-learn pipeline.
6. Evaluate on the held-out test set and save the metrics.
7. Load the saved pipeline in Streamlit for predictions.

The earlier report records approximately 96% accuracy on 7,821 test examples.
Your newly trained result can differ because it depends on the exact CSV files,
preprocessing, and train/test split. Report the value printed by your own run.

## Important limitation

The model learns language patterns from its labelled dataset. It does not
search the web, inspect the publisher, or verify claims against evidence.
A “Real” output is not proof that an article is true, and a “Fake” output is
not proof that it is false.
