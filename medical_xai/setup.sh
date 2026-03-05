#!/bin/bash
echo "================================================"
echo " MedXAI - Setup (Mac / Linux)"
echo "================================================"
echo ""
echo "[1/3] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo ""
echo "[2/3] Installing packages..."
pip install -r requirements.txt
echo ""
echo "[3/3] Creating folders..."
mkdir -p weights records
echo ""
echo "================================================"
echo " DONE. Now:"
echo " 1. Put dataset in:  data/chest_xray/"
echo " 2. Train model:     python train.py"
echo " 3. Run app:         streamlit run app.py"
echo "================================================"
