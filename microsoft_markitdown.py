import sys
from markitdown import MarkItDown
from pathlib import Path
import unicodedata
import re

def reverse_hebrew_chars(text):
    """
    Reverse Hebrew characters while keeping other characters in place.
    This is needed because Hebrew text is being stored in reverse order.
    """
    # Hebrew Unicode range
    hebrew_range = '\u0590-\u05FF'
    
    # Find all Hebrew character sequences
    hebrew_pattern = f'[{hebrew_range}]+'
    
    def reverse_hebrew_match(match):
        hebrew_text = match.group(0)
        # Reverse the Hebrew characters
        return hebrew_text[::-1]
    
    # Apply the reversal to Hebrew sequences
    result = re.sub(hebrew_pattern, reverse_hebrew_match, text)
    return result

def normalize_hebrew_text(text):
    """
    Normalize Hebrew text to ensure proper character ordering.
    """
    # First normalize Unicode
    normalized = unicodedata.normalize('NFC', text)
    
    # Check if there's Hebrew text
    if any('\u0590' <= char <= '\u05FF' for char in normalized):
        # Reverse Hebrew characters that are in wrong order
        fixed_text = reverse_hebrew_chars(normalized)
        
        # Add RTL markers for proper display
        return '\u202B' + fixed_text + '\u202C'
    else:
        return normalized

def main():
    input_file_path = Path(r"C:\Coding_projects\Hebrew-PDF-OCR-Interactive-Editor\input_data\3-5-2025-23-6_31.pdf")
    output_file_path = Path(r"C:\Coding_projects\Hebrew-PDF-OCR-Interactive-Editor\output_data\3-5-2025-23-6_31.md")

    print(f"Converting {input_file_path} to Markdown...")

    # Initialize MarkItDown. Set enable_plugins=True if you have plugins.
    md = MarkItDown()

    result = md.convert(input_file_path)
    
    # Fix Hebrew character ordering
    processed_content = normalize_hebrew_text(result.text_content)

    print("Processed content:")
    print(processed_content)

    # Write with UTF-8 BOM to ensure proper encoding detection
    with open(output_file_path, "w", encoding="utf-8-sig") as f:
        f.write(processed_content)
    
    print(f"Conversion completed! Output saved to: {output_file_path}")
    print("Note: The file includes RTL text direction markers for proper Hebrew display.")

if __name__ == "__main__":
    main()
