@echo off
echo ================================================
echo  MedXAI - Setup (Windows)
echo ================================================
echo.
echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate
echo.
echo [2/3] Installing packages (takes 5-10 mins)...
pip install -r requirements.txt
echo.
echo [3/3] Creating folders...
if not exist "weights" mkdir weights
if not exist "records" mkdir records
echo.
echo ================================================
echo  DONE. Now:
echo  1. Put dataset in:  data\chest_xray\
echo  2. Train model:     python train.py
echo  3. Run app:         streamlit run app.py
echo ================================================
pause
