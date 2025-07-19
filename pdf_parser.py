from transformers import AutoProcessor, AutoModelForVision2Seq
from pdf2image import convert_from_path
from PIL import Image
import os
import torch
from pathlib import Path

def ocr_pdf_with_qwen(file_path, poppler_path=None):
    """
    Performs OCR on a PDF file using the Qwen2.5-VL-7B model.
    """
    # Load the processor and model
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True
    )

    # Convert PDF to a list of images
    try:
        images = convert_from_path(file_path, poppler_path=poppler_path)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        print("Please ensure Poppler is installed and the path is correct.")
        return

    # Create a directory for temporary images
    temp_image_dir = "temp_images"
    if not os.path.exists(temp_image_dir):
        os.makedirs(temp_image_dir)

    # Process each image
    for i, image in enumerate(images):
        print(f"--- Processing Page {i+1} ---")

        # Save the image to a temporary file
        temp_image_path = os.path.join(temp_image_dir, f"image.png")
        image.save(temp_image_path)
        
        # Prepare the prompt for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_image_path},
                    {"type": "text", "text": "what do you see?"}
                ]
            }
        ]
        
        # Process the inputs
        try:
            input_ids = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
            attention_mask = torch.ones_like(input_ids)
            inputs = {"input_ids": input_ids.to(model.device), "attention_mask": attention_mask.to(model.device)}

            # Generate text
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(response)

        except Exception as e:
            print(f"An error occurred during model processing: {e}")
        finally:
            # Clean up the temporary image file
            os.remove(temp_image_path)

    # Clean up the temporary image directory
    if os.path.exists(temp_image_dir):
        os.rmdir(temp_image_dir)

def main():
    input_dir = "input_data"
    
    # Create input_data directory if it doesn't exist
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory: {input_dir}")
        print("Please add your PDF files to this directory and run the script again.")
        return
    
    # --- IMPORTANT ---
    # Set the path to your Poppler installation's 'bin' directory.
    # If Poppler is in your system's PATH, you can set this to None.
    # Example for Windows: r"C:\path\to\poppler-xx.xx.x\bin"
    poppler_installation_path = None # <-- SET THIS PATH

    # For now, let's just parse one file to test the OCR.
    filename = "3-5-2025-23-6_31.pdf"
    file_path = os.path.join(input_dir, filename)
    
    ocr_pdf_with_qwen(file_path, poppler_path=poppler_installation_path)

if __name__ == "__main__":
    main() 