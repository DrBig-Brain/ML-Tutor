import fitz
import re
from pathlib import Path
from typing import List,Dict

class PDFProcessor:
    def __init__(self, pdf_path:str):
        self.pdf_path = Path(pdf_path)
        self.doc = None

    def extract_text(self) -> str:
        self.doc = fitz.open(self.pdf_path)
        text = []

        for page_number in range(self.doc.page_count):
            page = self.doc[page_number]
            page_text = page.get_text()
            text.append(f"\n-- Page {page_number + 1} --\n")
            text.append(page_text)

        return "".join(text)
    
    @staticmethod
    def clean_text(text:str) -> str:
        text = re.sub(r'\n\s*\n','\n\n',text)
        text = re.sub(r'[^\x00-\x7F]+',' ',text)
        text = re.sub(r' +',' ',text)

        return text.strip()
    
    def process(self) -> str:
        raw_text = self.extract_text()
        cleaned_text = self.clean_text(raw_text)
        return cleaned_text
    