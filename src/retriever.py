import json

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class EcoSortRetriever:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.index = faiss.read_index(
            "data/vector_store/index.faiss"
        )

        with open(
            "data/vector_store/chunks.json",
            "r",
            encoding="utf-8"
        ) as f:

            self.chunks = json.load(f)


    def search(
        self,
        query,
        top_k=3
    ):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            chunk = self.chunks[index]

            results.append({
                "score": float(score),
                "text": chunk["text"],
                "filename": chunk["filename"],
                "page": chunk["page"]
            })

        return results