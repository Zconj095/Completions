import PyPDF2
import os
from pathlib import Path

def read_pdf(file_path, page_num=None):
    """
    Read PDF file and extract text from specified page or all pages.
    
    Args:
        file_path (str): Path to the PDF file
        page_num (int, optional): Specific page number to read (0-indexed). If None, reads all pages.
    
    Returns:
        str: Extracted text
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        with open(file_path, "rb") as pdf_file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Get number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Total pages: {num_pages}")
            
            extracted_text = ""
            
            if page_num is not None:
                # Read specific page
                if 0 <= page_num < num_pages:
                    page = pdf_reader.pages[page_num]
                    extracted_text = page.extract_text()
                    print(f"Reading page {page_num + 1}")
                else:
                    raise IndexError(f"Page {page_num} out of range (0-{num_pages-1})")
            else:
                # Read all pages
                print("Reading all pages...")
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    extracted_text += f"\n--- Page {i + 1} ---\n{text}\n"
            
            return extracted_text.strip()
            
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Usage
if __name__ == "__main__":
    pdf_path = r"C:\PDF_Files\Unlocking_the_Brain_Coding.pdf"
    
    # Read first page only
    text = read_pdf(pdf_path, page_num=0)
    if text:
        print("Extracted text:")
        print(text)
    
    # Uncomment to read all pages
    # all_text = read_pdf(pdf_path)
    # if all_text:
    #     print(all_text)
