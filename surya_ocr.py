from PIL import Image
from pdf2image import convert_from_path
import torch
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import os
import argparse

def ocr_pdf_with_surya(file_path, poppler_path=None):
    """
    Performs OCR on a PDF file using the surya library.
    """
    # 1. Convert PDF to images
    print("Converting PDF to images...")
    try:
        images = convert_from_path(file_path, poppler_path=poppler_path)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        print("Please ensure Poppler is installed and the path is correct.")
        return
    print(f"Converted {len(images)} pages.")

    # 2. Load predictors
    print("Loading models...")
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()
    print("Models loaded.")

    # 3. Process images with surya OCR
    print("Running OCR on images...")
    predictions = rec_predictor(images, det_predictor=det_predictor)
    print("OCR complete.")

    # 4. Print and return results
    all_text = []
    for i, pred in enumerate(predictions):
        print(f"--- Page {i+1} Text ---")
        page_text = "\n".join([line.text for line in pred.text_lines])
        print(page_text)
        all_text.append(page_text)

    return all_text

def main():
    parser = argparse.ArgumentParser(description="OCR a PDF file using surya.")
    parser.add_argument("--filename", type=str, default="3-5-2025-23-6_31.pdf", help="Name of the PDF file in the input_data directory.")
    parser.add_argument("--poppler_path", type=str, default=None, help="Path to your Poppler installation's 'bin' directory.")
    args = parser.parse_args()

    input_dir = "input_data"

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory: {input_dir}")
        print("Please add your PDF files to this directory and run the script again.")
        return

    file_path = os.path.join(input_dir, args.filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    ocr_pdf_with_surya(file_path, poppler_path=args.poppler_path)

if __name__ == "__main__":
    main() 