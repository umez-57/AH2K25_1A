
import os
import json
from pdf_processor import extract_text_with_layout
from heading_extractor import extract_title, assign_levels, clean_text
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def evaluate_pdf(pdf_path, ground_truth_path):
    extracted_data = extract_text_with_layout(pdf_path)
    predicted_outline = assign_levels(extracted_data)
    predicted_title = extract_title(extracted_data)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    true_outline = ground_truth.get("outline", [])
    true_title = ground_truth.get("title", "")

    # Evaluate Title
    cleaned_predicted_title = clean_text(predicted_title)
    cleaned_true_title = clean_text(true_title)
    title_match = 1.0 if cleaned_predicted_title == cleaned_true_title else 0.0

    # Convert outlines to a set of (cleaned_text, page) tuples for easier comparison
    predicted_set = {(clean_text(item["text"]), item["page"]) for item in predicted_outline}
    
    # For true_set, we need to consider potential page number discrepancies.
    # The ground truth seems to be 0-indexed in some cases, and 1-indexed in others.
    # We will assume our `pdf_processor` extracts 1-based page numbers.
    # So, we will normalize the ground truth to be 1-based for comparison.
    flexible_true_set = set()
    for item in true_outline:
        cleaned_text = clean_text(item["text"])
        original_page = item["page"]
        
        # Heuristic: if ground truth page is 0, assume it should be 1. Otherwise, use the original page number.
        # This is a common discrepancy. Also consider +/- 1 page for flexibility.
        normalized_page = original_page + 1 if original_page == 0 else original_page

        # Add the normalized page number and its neighbors
        flexible_true_set.add((cleaned_text, normalized_page))
        if normalized_page > 1:
            flexible_true_set.add((cleaned_text, normalized_page - 1))
        flexible_true_set.add((cleaned_text, normalized_page + 1))

    # True Positives: predicted as heading and is actually a heading (considering flexible page numbers)
    tp = len(predicted_set.intersection(flexible_true_set))

    # False Positives: predicted as heading but is not actually a heading
    # This needs to be calculated against the *original* true_set to avoid penalizing for flexible matching
    original_true_set = set()
    for item in true_outline:
        cleaned_text = clean_text(item["text"])
        original_page = item["page"]
        normalized_page = original_page + 1 if original_page == 0 else original_page
        original_true_set.add((cleaned_text, normalized_page))

    fp = len(predicted_set - original_true_set)

    # False Negatives: not predicted as heading but is actually a heading
    # This also needs to be calculated against the *original* true_set
    fn = len(original_true_set - predicted_set)

    # Calculate metrics
    outline_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    outline_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    outline_f1 = 2 * (outline_precision * outline_recall) / (outline_precision + outline_recall) if (outline_precision + outline_recall) > 0 else 0

    # Combine scores (simple average for now, can be weighted later)
    if not cleaned_true_title and not cleaned_predicted_title:
        overall_precision = outline_precision
        overall_recall = outline_recall
        overall_f1 = outline_f1
    else:
        overall_precision = (outline_precision + title_match) / 2
        overall_recall = (outline_recall + title_match) / 2
        overall_f1 = (outline_f1 + title_match) / 2

    return overall_precision, overall_recall, overall_f1

def evaluate_all_pdfs(pdf_dir, ground_truth_dir):
    all_precision = []
    all_recall = []
    all_f1 = []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        json_file = pdf_file.replace(".pdf", ".json")
        ground_truth_path = os.path.join(ground_truth_dir, json_file)

        if not os.path.exists(ground_truth_path):
            print(f"Warning: Ground truth JSON not found for {pdf_file}. Skipping evaluation.")
            continue

        precision, recall, f1 = evaluate_pdf(pdf_path, ground_truth_path)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        print(f"Evaluated {pdf_file}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

    avg_precision = np.mean(all_precision) if all_precision else 0
    avg_recall = np.mean(all_recall) if all_recall else 0
    avg_f1 = np.mean(all_f1) if all_f1 else 0

    print("\n--- Overall Averages ---")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")

    return avg_precision, avg_recall, avg_f1

if __name__ == "__main__":
    pdf_directory = "/home/ubuntu/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs"
    ground_truth_directory = "/home/ubuntu/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/outputs"
    
    evaluate_all_pdfs(pdf_directory, ground_truth_directory)


