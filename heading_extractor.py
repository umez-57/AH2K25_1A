
import numpy as np
import re
import joblib
import pandas as pd

# Load the trained XGBoost model
MODEL_PATH = "xgboost_heading_model.joblib"
try:
    XGB_MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError:
    XGB_MODEL = None
    print("Warning: XGBoost model not found. Running with heuristics only.")

def clean_text(t: str) -> str:
    # Only strip leading/trailing whitespace and standardize internal whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_features_for_prediction(line: dict) -> list:
    return [
        line["font_size"],
        1 if line["is_bold"] else 0,
        line["bbox"][0], # x0
        line["bbox"][1], # y0
        line["bbox"][2], # x1
        line["bbox"][3], # y1
        line["page_width"],
        line["page_height"]
    ]

def assign_levels(lines: list[dict]) -> list[dict]:
    if not lines:
        return []

    lines_by_page = {}
    for line in lines:
        page_num = line["page"]
        if page_num not in lines_by_page:
            lines_by_page[page_num] = []
        lines_by_page[page_num].append(line)

    median_font_sizes = {}
    for page_num, page_lines in lines_by_page.items():
        font_sizes = [l["font_size"] for l in page_lines]
        if font_sizes:
            median_font_sizes[page_num] = np.median(font_sizes)
        else:
            median_font_sizes[page_num] = 0

    potential_headings = []
    for line in lines:
        cleaned_line_text = clean_text(line["text"])
        if not cleaned_line_text: # Skip empty or whitespace-only lines
            continue

        is_potential_heading = False

        if XGB_MODEL:
            features = extract_features_for_prediction(line)
            feature_df = pd.DataFrame([features], columns=["font_size", "is_bold", "x0", "y0", "x1", "y1", "page_width", "page_height"])
            prediction = XGB_MODEL.predict(feature_df)[0]
            if prediction == 1:
                is_potential_heading = True
        else:
            # Fallback to heuristics if model is not loaded
            if line["is_bold"] and line["font_size"] >= median_font_sizes.get(line["page"], 0) * 1.2:
                is_potential_heading = True
            elif re.match(r'^\d+(\.\d+)*\s', cleaned_line_text):
                is_potential_heading = True
            elif line["font_size"] > median_font_sizes.get(line["page"], 0) * 1.5:
                is_potential_heading = True

        # Filter out short, non-descriptive lines that might be false positives (like dates, page numbers, etc.)
        # Also filter out lines that are mostly numbers or symbols unless they are clearly numbered headings
        if is_potential_heading:
            # Heuristic to filter out short, non-descriptive lines that are not numbered headings
            if len(cleaned_line_text.split()) < 2 and not re.match(r'^\d+(\.\d+)*$', cleaned_line_text):
                is_potential_heading = False
            # Further filter out lines that are purely numeric or contain only a few alphabets and many non-alphanumeric chars
            elif not re.search(r'[a-zA-Z]{3,}', cleaned_line_text) and not re.match(r'^\d+(\.\d+)*$', cleaned_line_text):
                is_potential_heading = False

        if is_potential_heading:
            potential_headings.append(line)

    # Sort potential headings by font size (descending) and then by page and bbox for original order
    potential_headings.sort(key=lambda x: (-x["font_size"], x["page"], x["bbox"][1]))

    # Group potential headings by page and then by font size to determine hierarchy
    grouped_by_page_and_font = {}
    for ph in potential_headings:
        page_key = ph["page"]
        font_size_key = ph["font_size"]
        if page_key not in grouped_by_page_and_font:
            grouped_by_page_and_font[page_key] = {}
        if font_size_key not in grouped_by_page_and_font[page_key]:
            grouped_by_page_and_font[page_key][font_size_key] = []
        grouped_by_page_and_font[page_key][font_size_key].append(ph)

    outline = []
    for page_num in sorted(grouped_by_page_and_font.keys()):
        font_sizes_on_page = sorted(grouped_by_page_and_font[page_num].keys(), reverse=True)
        
        size_to_level_on_page = {}
        if len(font_sizes_on_page) >= 1: size_to_level_on_page[font_sizes_on_page[0]] = "H1"
        if len(font_sizes_on_page) >= 2: size_to_level_on_page[font_sizes_on_page[1]] = "H2"
        if len(font_sizes_on_page) >= 3: size_to_level_on_page[font_sizes_on_page[2]] = "H3"

        for font_size in font_sizes_on_page:
            for h in grouped_by_page_and_font[page_num][font_size]:
                cleaned_text_h = clean_text(h["text"])
                
                lvl = size_to_level_on_page.get(h["font_size"], "H3") # Default level

                # Override level assignment based on numbering patterns for more accuracy
                if re.match(r'^\d+\.\d+\.\d+', cleaned_text_h):
                    lvl = "H3"
                elif re.match(r'^\d+\.\d+', cleaned_text_h):
                    lvl = "H2"
                elif re.match(r'^\d+\.', cleaned_text_h):
                    lvl = "H1"
                
                outline.append({"level": lvl, "text": cleaned_text_h, "page": h["page"], "bbox": h["bbox"]})

    # Remove duplicates based on level, text, and page, then sort by page and y-coordinate
    seen_tuples = set()
    final_outline = []
    for item in outline:
        item_tuple = (item["level"], item["text"], item["page"])
        if item_tuple not in seen_tuples:
            final_outline.append(item)
            seen_tuples.add(item_tuple)
    
    final_outline.sort(key=lambda x: (x["page"], x["bbox"][1])) 

    return final_outline

def extract_title(lines: list[dict]) -> str:
    if not lines:
        return ""

    first_page_lines = [l for l in lines if l["page"] == 0]
    if not first_page_lines:
        return ""

    first_page_lines.sort(key=lambda x: (-x["font_size"], x["bbox"][1]))

    if first_page_lines:
        candidate_title = first_page_lines[0]
        cleaned_title_text = clean_text(candidate_title["text"])
        
        if len(cleaned_title_text.split()) > 2 and not re.match(r'^\d+(\.\d+)*\s*$', cleaned_title_text):
            return cleaned_title_text
        
        for line in first_page_lines[1:]:
            cleaned_line_text = clean_text(line["text"])
            if len(cleaned_line_text.split()) > 2 and not re.match(r'^\d+(\.\d+)*\s*$', cleaned_line_text):
                return cleaned_line_text

    return ""


if __name__ == "__main__":
    from pdf_processor import extract_text_with_layout

    pdf_file = "/home/ubuntu/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/file01.pdf"
    extracted_data = extract_text_with_layout(pdf_file)

    title = extract_title(extracted_data)
    outline = assign_levels(extracted_data)

    print(f"Title: {title}")
    print("Outline:")
    for item in outline:
        print(item)


