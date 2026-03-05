# MedXAI — Explainable Medical Image Classification
### MSc / PhD Research Project | Pneumonia Detection from Chest X-Rays

---

## File Structure

```
medical_xai/
│
├── app.py                        ← Run this to start the app
├── train.py                      ← Train the model
├── requirements.txt              ← All dependencies
├── setup.bat                     ← Windows one-click setup
├── setup.sh                      ← Mac/Linux one-click setup
│
├── .streamlit/
│   └── config.toml               ← App theme and settings
│
├── models/
│   ├── __init__.py
│   └── fusion_model.py           ← ResNet18 + classifier
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py          ← Image transforms
│   ├── image_quality.py          ← Quality checker (blur, noise, spots)
│   ├── treatment_protocol.py     ← Treatment plans by severity
│   ├── report_generator.py       ← PDF report generator
│   └── feedback_dataset.py       ← Clinical summary cards
│
├── explainability/
│   ├── __init__.py
│   ├── gradcam.py                ← Grad-CAM heatmap
│   ├── lime_exp.py               ← LIME superpixels
│   ├── shap_exp.py               ← SHAP attribution
│   └── xray_annotator.py         ← Lung zone bounding boxes
│
├── evaluation/
│   ├── __init__.py
│   └── escore.py                 ← Custom E-Score metric
│
├── data/
│   └── chest_xray/               ← PUT KAGGLE DATASET HERE
│       ├── train/NORMAL/
│       ├── train/PNEUMONIA/
│       ├── val/NORMAL/
│       ├── val/PNEUMONIA/
│       ├── test/NORMAL/
│       └── test/PNEUMONIA/
│
├── weights/                      ← Model saved here after training
└── records/                      ← Patient CSV logs saved here
```

---

## Setup (Windows)

**Step 1 — Requires Python 3.10**
Download: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
During install tick: Add Python to PATH

**Step 2 — Open PowerShell in the medical_xai folder**
```
cd "C:\path\to\medical_xai"
```

**Step 3 — Create virtual environment**
```
python -m venv venv
venv\Scripts\activate
```

**Step 4 — Install packages**
```
pip install -r requirements.txt
```

**Step 5 — Download dataset**
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Unzip into data/chest_xray/

**Step 6 — Train model**
```
python train.py
```

**Step 7 — Run app**
```
streamlit run app.py
```
Open browser at: http://localhost:8501

---

## If streamlit command not found
```
python -m streamlit run app.py
```

## If port 8501 is busy
```
streamlit run app.py --server.port 8502
```

---

## Dataset
- Name: Chest X-Ray Images (Pneumonia)
- URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Classes: NORMAL (1583 images) | PNEUMONIA (4273 images)
- Format: JPEG

---

## Expected Results
- Test Accuracy: 92-95%
- Pneumonia Recall: 96-98%
- Training Time (CPU): 2-4 hours / 15 epochs

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| DLL load failed | Wrong Python version. Use Python 3.10 only |
| streamlit not found | Use: python -m streamlit run app.py |
| File does not exist: app.py | cd into medical_xai folder first |
| Port not available | Add --server.port 8502 |
| Long path error | Enable Windows long paths or use Python 3.10 installer |
| ModuleNotFoundError | pip install -r requirements.txt |

---

*For Research Use Only. Not a medical device.*
