import faiss
import numpy as np


class EcoSortRAG:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = []

        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        self.index = faiss.IndexFlatIP(384)

    def add_documents(self, documents):

        vectors = self.embedding_model.encode(documents)

        self.index.add(
            np.asarray(vectors, dtype="float32")
        )

        self.documents.extend(documents)

    def search(self, query, k=3):

        if not self.documents:
            return []

        query_vector = self.embedding_model.encode(
            [query]
        )

        scores, indices = self.index.search(
            np.asarray(query_vector, dtype="float32"),
            min(k, len(self.documents))
        )

        results = []

        for index in indices[0]:

            if index != -1:
                results.append(
                    self.documents[index]
                )

        return results