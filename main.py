#!/usr/bin/env python3
"""
PDF Heading Extraction Solution for Adobe India Hackathon 2025 - Challenge 1a

This script extracts structured outlines (title and headings) from PDF files.
It uses a combination of heuristic rules and machine learning to identify headings.

Usage:
    python main.py input_dir output_dir

Where:
    input_dir: Directory containing PDF files to process
    output_dir: Directory where JSON output files will be saved
"""

import os
import sys
import json
import time
from pdf_processor import extract_text_with_layout
from heading_extractor import extract_title, assign_levels

def process_pdf(pdf_path: str) -> dict:
    """
    Process a single PDF file and extract title and outline.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing title and outline
    """
    try:
        # Extract text with layout information
        extracted_data = extract_text_with_layout(pdf_path)
        
        # Extract title
        title = extract_title(extracted_data)
        
        # Extract outline (headings)
        outline = assign_levels(extracted_data)
        
        # Format outline for output
        formatted_outline = []
        for item in outline:
            formatted_outline.append({
                "level": item["level"],
                "text": item["text"],
                "page": item["page"]
            })
        
        return {
            "title": title,
            "outline": formatted_outline
        }
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return {
            "title": "",
            "outline": []
        }

def main():
    """Main function to process all PDFs in input directory."""
    if len(sys.argv) != 3:
        print("Usage: python main.py input_dir output_dir")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    start_time = time.time()
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_file = pdf_file.replace('.pdf', '.json')
        output_path = os.path.join(output_dir, output_file)
        
        print(f"Processing {pdf_file}...")
        
        # Process the PDF
        result = process_pdf(pdf_path)
        
        # Save the result
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved output to {output_file}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Average time per PDF: {total_time/len(pdf_files):.2f} seconds")

if __name__ == "__main__":
    main()

