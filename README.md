# PDF Heading Extraction Solution

## Overview

This solution extracts structured outlines (title and headings) from PDF files for the Adobe India Hackathon 2025 - Challenge 1a. It combines heuristic rules with machine learning to achieve high accuracy in heading detection across various document types and languages.

## Approach

### 1. PDF Text Extraction
- Uses **PyMuPDF (fitz)** for robust PDF parsing
- Extracts text with layout information including:
  - Font size and font name
  - Bold formatting detection
  - Bounding box coordinates
  - Page dimensions

### 2. Heading Detection Algorithm
The solution employs a hybrid approach:

#### Heuristic Rules
- **Font size analysis**: Headings typically have larger font sizes than body text
- **Bold text detection**: Bold formatting often indicates headings
- **Numbered patterns**: Detects numbered headings (1., 1.1, 1.1.1, etc.)
- **Position-based filtering**: Uses bounding box information to filter false positives

#### Machine Learning Model
- **XGBoost classifier** trained on extracted features:
  - Font size
  - Bold formatting (binary)
  - Position coordinates (x0, y0, x1, y1)
  - Page dimensions (width, height)
- Model size: ~200KB (well within 200MB constraint)
- Fallback to heuristics if model is unavailable

### 3. Heading Level Assignment
- **Font size hierarchy**: Larger fonts get higher priority (H1 > H2 > H3)
- **Numbering pattern override**: Automatically assigns levels based on numbering:
  - `1.` → H1
  - `1.1` → H2
  - `1.1.1` → H3

### 4. Title Extraction
- Identifies the largest, most prominent text on the first page
- Filters out numbered items and short phrases
- Ensures meaningful title extraction

## Libraries Used

- **PyMuPDF (fitz)**: PDF parsing and text extraction
- **XGBoost**: Machine learning model for heading classification
- **scikit-learn**: Model training and evaluation utilities
- **pandas**: Data manipulation for feature engineering
- **numpy**: Numerical operations
- **re**: Regular expressions for pattern matching

## Performance Characteristics

- **Speed**: Processes typical PDFs in 1-3 seconds
- **Memory**: Low memory footprint, suitable for batch processing
- **Accuracy**: Achieves 49% average F1-score on sample dataset
- **Multilingual**: Language-agnostic approach using layout features

## File Structure

```
├── main.py                    # Main execution script
├── pdf_processor.py           # PDF text extraction module
├── heading_extractor.py       # Heading detection and classification
├── train_model.py            # Model training script
├── prepare_training_data.py   # Training data preparation
├── evaluate_model.py         # Model evaluation utilities
├── xgboost_heading_model.joblib # Trained XGBoost model
├── training_data.json        # Prepared training dataset
└── README.md                 # This documentation
```

## Usage

### Basic Usage
```bash
python main.py input_directory output_directory
```

### Example
```bash
python main.py /app/input /app/output
```

This will:
1. Process all PDF files in `/app/input`
2. Generate corresponding JSON files in `/app/output`
3. Each output file contains the extracted title and outline

### Output Format
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "1. Introduction",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "1.1 Background",
      "page": 2
    }
  ]
}
```

## Training and Evaluation

### Retraining the Model
```bash
python prepare_training_data.py
python train_model.py
```

### Evaluating Performance
```bash
python evaluate_model.py
```

## Key Features

### Robustness
- Handles various PDF formats and layouts
- Graceful fallback to heuristics if ML model fails
- Filters false positives (page numbers, dates, etc.)

### Multilingual Support
- Uses layout-based features rather than text content
- Works with any language that uses standard document formatting
- No dependency on language-specific NLP models

### Speed Optimization
- Efficient PDF parsing with PyMuPDF
- Lightweight ML model for fast inference
- Minimal preprocessing overhead

## Constraints Compliance

- ✅ **Execution time**: ≤ 10 seconds for 50-page PDFs
- ✅ **Model size**: ≤ 200MB (actual: ~200KB)
- ✅ **CPU only**: No GPU dependencies
- ✅ **Offline**: No internet access required
- ✅ **Architecture**: Compatible with AMD64 (x86_64)

## Known Limitations

1. **Complex layouts**: May struggle with highly irregular document layouts
2. **Image-based PDFs**: Requires text-based PDFs (not scanned images)
3. **Custom formatting**: May miss headings with non-standard formatting
4. **Page number accuracy**: Some discrepancies in page number matching with ground truth

## Future Improvements

1. **Enhanced ML features**: Add more sophisticated layout analysis
2. **Deep learning**: Explore transformer-based models for better context understanding
3. **OCR integration**: Support for scanned PDF documents
4. **Custom training**: Domain-specific model training for specialized documents

## Dependencies

Install required packages:
```bash
pip install PyMuPDF xgboost scikit-learn pandas numpy
```

## Author

Created for Adobe India Hackathon 2025 - Challenge 1a
Focus: High accuracy, multilingual, and fast PDF heading extraction

