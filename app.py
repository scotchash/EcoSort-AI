import streamlit as st

from src.retriever import EcoSortRetriever
from src.llm import EcoSortLLM


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide"
)


# --------------------------------------------------
# Load EcoSort system
# --------------------------------------------------

@st.cache_resource
def load_system():

    retriever = EcoSortRetriever()
    llm = EcoSortLLM()

    return retriever, llm


retriever, llm = load_system()


# --------------------------------------------------
# User Interface
# --------------------------------------------------

st.title("♻️ EcoSort AI")

st.subheader(
    "AI-Powered Waste Segregation & "
    "Sustainable Disposal Assistant"
)

st.write(
    "Ask a question about waste management "
    "using our official-source RAG knowledge base."
)

st.divider()


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_input(
    "🔎 Ask about a waste item",
    placeholder="Example: Where should I dispose of an old mobile phone?"
)


# --------------------------------------------------
# Analyze
# --------------------------------------------------

if st.button("🔍 Analyze"):

    if not question:

        st.warning(
            "Please enter a question."
        )

    else:

        # ------------------------------------------
        # Retrieve relevant documents
        # ------------------------------------------

        with st.spinner(
            "Searching official knowledge sources..."
        ):

            documents = retriever.search(
                question,
                top_k=3
            )


        # ------------------------------------------
        # Generate grounded answer
        # ------------------------------------------

        if documents:

            with st.spinner(
                "Generating grounded response..."
            ):

                answer = llm.answer(
                    question,
                    documents
                )


            # --------------------------------------
            # Display answer
            # --------------------------------------

            st.subheader(
                "♻️ AI Analysis"
            )

            st.write(answer)


            # --------------------------------------
            # Display sources
            # --------------------------------------

            st.subheader(
                "📚 Retrieved Sources"
            )

            for document in documents:

                st.write(
                    f"**{document['filename']}** "
                    f"— Page {document['page']}"
                )

        else:

            st.warning(
                "No relevant information was found "
                "in the knowledge base."
            )