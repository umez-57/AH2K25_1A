
import fitz # PyMuPDF

def extract_text_with_layout(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_data = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height
        # Use get_text("dict") for detailed block information
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    # Reconstruct text from spans to ensure correct line text
                    full_line_text = "".join(span["text"] for span in line["spans"]).strip()
                    if not full_line_text: # Skip empty lines
                        continue
                    
                    # Use the bbox of the entire line for more accurate position
                    line_bbox = line["bbox"]
                    # Determine if the line is bold by checking if any span is bold
                    is_bold = any("bold" in span["font"].lower() for span in line["spans"])

                    extracted_data.append({
                        "text": full_line_text,
                        "font_size": line["spans"][0]["size"], # Use first span's font size as representative
                        "font_name": line["spans"][0]["font"], # Use first span's font name
                        "bbox": line_bbox,
                        "page": page_num, # 1-based page number
                        "page_width": page_width,
                        "page_height": page_height,
                        "is_bold": is_bold
                    })
    return extracted_data

if __name__ == "__main__":
    # Example usage (for testing purposes)
    pdf_file = "/home/ubuntu/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/file01.pdf"
    data = extract_text_with_layout(pdf_file)
    for item in data[:20]: # Print first 20 items for a quick check
        print(item)


