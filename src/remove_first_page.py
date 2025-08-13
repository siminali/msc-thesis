#!/usr/bin/env python3
"""
Remove First Page from PDF

This script removes the first page from the corrected comprehensive report PDF.
"""

import os
from PyPDF2 import PdfReader, PdfWriter

def remove_first_page(input_path, output_path):
    """Remove the first page from a PDF file."""
    try:
        # Read the input PDF
        reader = PdfReader(input_path)
        
        # Check if PDF has pages
        if len(reader.pages) == 0:
            print("❌ PDF has no pages!")
            return False
        
        print(f"📄 Input PDF has {len(reader.pages)} pages")
        
        # Create a PDF writer
        writer = PdfWriter()
        
        # Add all pages except the first one (index 0)
        for i in range(1, len(reader.pages)):
            writer.add_page(reader.pages[i])
            print(f"✅ Added page {i+1} (was page {i+1} in original)")
        
        # Write the output PDF
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"✅ Successfully removed first page!")
        print(f"📄 Output PDF: {output_path}")
        print(f"📊 Pages: {len(reader.pages) - 1} (was {len(reader.pages)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return False

def main():
    """Main function."""
    input_path = "results/comprehensive_model_comparison_report_corrected.pdf"
    output_path = "results/comprehensive_model_comparison_report_corrected_no_first_page.pdf"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    
    print(f"🚀 Removing first page from: {input_path}")
    
    # Remove first page
    success = remove_first_page(input_path, output_path)
    
    if success:
        print(f"\n✅ Task completed successfully!")
        print(f"📄 Original file: {input_path}")
        print(f"📄 Modified file: {output_path}")
    else:
        print(f"\n❌ Task failed!")

if __name__ == "__main__":
    main()
