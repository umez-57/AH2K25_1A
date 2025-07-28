import os
import json
from pdf_processor import extract_text_with_layout
from heading_extractor import clean_text

def prepare_data(pdf_dir, json_dir):
    training_data = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        json_file = pdf_file.replace(".pdf", ".json")
        json_path = os.path.join(json_dir, json_file)

        if not os.path.exists(json_path):
            print(f"Warning: Ground truth JSON not found for {pdf_file}. Skipping.")
            continue

        extracted_lines = extract_text_with_layout(pdf_path)
        with open(json_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        gt_outline_texts = {clean_text(item["text"]) for item in ground_truth.get("outline", [])}

        for line in extracted_lines:
            cleaned_line_text = clean_text(line["text"])
            # Create features for each line
            features = {
                "text": cleaned_line_text,
                "font_size": line["font_size"],
                "is_bold": line["is_bold"],
                "page": line["page"],
                "bbox": line["bbox"],
                "page_width": line["page_width"],
                "page_height": line["page_height"],
                # Add more features here later, like relative position, indentation, etc.
            }
            # Label as heading or not
            is_heading = 1 if cleaned_line_text in gt_outline_texts else 0
            training_data.append({"features": features, "label": is_heading})

    return training_data

if __name__ == "__main__":
    pdf_directory = "/home/ubuntu/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs"
    json_directory = "/home/ubuntu/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/outputs"
    
    data = prepare_data(pdf_directory, json_directory)
    print(f"Prepared {len(data)} data points.")
    # You can save this data to a file for later use
    with open("training_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Training data saved to training_data.json")

