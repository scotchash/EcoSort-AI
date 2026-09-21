import json
from pathlib import Path


INPUT_FILE = Path(
    "knowledge_base/processed/documents.json"
)

OUTPUT_FILE = Path(
    "knowledge_base/processed/chunks.json"
)


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def chunk_text(text):

    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        documents = json.load(f)

    chunks = []

    chunk_id = 0

    for document in documents:

        document_chunks = chunk_text(
            document["text"]
        )

        for chunk in document_chunks:

            chunks.append({
                "chunk_id": chunk_id,
                "filename": document["filename"],
                "page": document["page"],
                "text": chunk
            })

            chunk_id += 1

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Created {len(chunks)} chunks."
    )


if __name__ == "__main__":
    main()