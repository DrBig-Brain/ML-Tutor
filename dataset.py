import fitz
from tqdm import tqdm

def pdf_to_text(pdf_path, txt_path):
    reader = fitz.open(pdf_path)

    with open(txt_path,"w", encoding="utf-8") as f:
        for page in tqdm(reader, desc="Converting PDF", unit="page"):
            text = page.get_text()
            if text:
                f.write(text)
                f.write("\n\n")

if __name__ == "__main__":
    pdf_path = "dataset/dataset.pdf"
    txt_path = "dataset/dataset.txt"
    pdf_to_text(pdf_path,txt_path)
    print("successfully converted to txt")