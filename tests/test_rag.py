from src.embeddings import EmbeddingModel
from src.rag import EcoSortRAG


documents = [
    """
    E-waste includes electronic equipment such as mobile phones,
    computers, keyboards, chargers and batteries. E-waste should
    not be mixed with ordinary household waste.
    """,

    """
    Wet waste includes biodegradable materials such as vegetable
    peels, fruit peels, food leftovers, tea leaves and garden waste.
    """,

    """
    Dry recyclable waste can include paper, cardboard, plastic
    containers, plastic bottles, metal cans and glass bottles.
    """,

    """
    Hazardous household materials may require special handling
    and should not be mixed with ordinary recyclable waste.
    """
]


embedding_model = EmbeddingModel()

rag = EcoSortRAG(embedding_model)

rag.add_documents(documents)


question = "Where should an old mobile phone go?"

results = rag.search(question, k=2)


print("\n===== RAG RESULTS =====\n")

for i, result in enumerate(results, start=1):

    print(f"--- Result {i} ---")
    print(result)
    print()