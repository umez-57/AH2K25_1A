# Use a Python 3.10 base so all wheels (NumPy, PyMuPDF, scikit-learn, xgboost, pandas) are pre-built.
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install deps offline-friendly
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime files
COPY main.py pdf_processor.py heading_extractor.py xgboost_heading_model.joblib .

# Entry point
CMD ["python", "main.py", "/app/input", "/app/output"]
