from pathlib import Path
from pypdf import PdfReader
import json


DOCUMENT_DIR = Path("knowledge_base/documents")
OUTPUT_FILE = Path("knowledge_base/processed/documents.json")


def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def main():

    all_documents = []

    for pdf_path in DOCUMENT_DIR.glob("*.pdf"):

        print(f"Processing: {pdf_path.name}")

        pages = extract_pdf_text(pdf_path)

        for page in pages:

            all_documents.append({
                "filename": pdf_path.name,
                "page": page["page"],
                "text": page["text"]
            })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nSaved {len(all_documents)} pages to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()