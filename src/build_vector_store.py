import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path(
    "knowledge_base/processed/chunks.json"
)

VECTOR_DIR = Path(
    "data/vector_store"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        str(
            VECTOR_DIR / "index.faiss"
        )
    )

    with open(
        VECTOR_DIR / "chunks.json",
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
        f"Stored {len(chunks)} vectors."
    )


if __name__ == "__main__":
    main()