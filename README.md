# Radar Signal Classifier — Random Forest

A live ML project applying **Random Forest** to classify radar returns from the ionosphere
as **good** (structure detected) or **bad** (signal passed through) — built as part of a
DRDO AI Engineer internship task.

## Dataset
**Johns Hopkins University Ionosphere Database** (UCI ML Repository)
- 351 radar returns, collected by a phased-array system in Goose Bay, Labrador
- 34 continuous features per sample (autocorrelation values across 17 pulse numbers)
- Binary label: good / bad

## Why Random Forest
- Handles high-dimensional continuous data without much preprocessing
- Gives feature importance — shows which pulse signals matter most for classification
- Ensemble of trees (bagging + random feature selection) reduces overfitting vs a single tree

## Results
- **Test accuracy: ~93%** (200-tree Random Forest, 75/25 stratified train/test split)
- ROC-AUC and confusion matrix included in the app

## Project structure
```
radar_rf/
├── ionosphere.csv          # dataset
├── train_model.py          # training + evaluation script
├── app.py                  # Streamlit live app
├── rf_model.pkl             # trained model (generated)
├── feature_reference.csv    # feature ranges/samples for the app UI (generated)
├── confusion_matrix.png     # generated plot
├── feature_importance.png   # generated plot
├── roc_curve.png             # generated plot
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
python train_model.py     # trains model, saves rf_model.pkl + plots
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo
2. Go to https://share.streamlit.io → "New app" → select repo → main file `app.py`
3. Deploy — you get a public link in ~2 minutes

## App features
- **Live Prediction tab** — load a random real sample, tweak top signal features with sliders, get instant prediction + confidence
- **Model Performance tab** — confusion matrix, ROC curve, feature importance
- **About tab** — project write-up
