# Sleep Disruption ML Ops Project

This version converts the notebook workflow into a repeatable ML Ops-style pipeline.

## What it does

1. Loads a local CSV file from `data/raw_data.csv`
2. Creates a binary target column called `sleep_disrupted`
3. Selects the same feature columns from the notebook:
   - `age`
   - `gender`
   - `daily_gaming_hours`
   - `game_genre`
   - `primary_game`
   - `gaming_platform`
4. Handles missing values using median/mode imputation
5. Encodes categorical columns using saved category mappings
6. Applies `RobustScaler`
7. Splits the data into train/test sets
8. Trains a balanced Random Forest model
9. Evaluates the model at threshold `0.50`
10. Tunes the threshold from `0.30` to `0.70`
11. Runs `GridSearchCV` for Random Forest model selection
12. Saves the best model, scaler, mappings, threshold, and metrics

## Folder structure

```text
sleep_mloops_project/
├── app.py
├── requirements.txt
├── data/
│   └── raw_data.csv
├── models/
└── src/
    ├── ingest.py
    ├── preprocess.py
    ├── train.py
    └── predict.py
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Put your dataset CSV here:

```text
data/raw_data.csv
```

Then train the model:

```bash
python src/train.py
```

Run the app:

```bash
python app.py
```
