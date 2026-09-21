import os

import streamlit as st

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference


# Load local .env when running on your computer
load_dotenv()


def get_secret(name):

    # 1. Try environment variable / local .env
    value = os.getenv(name)

    if value:
        return value

    # 2. Try Streamlit Cloud Secrets
    try:
        return st.secrets[name]
    except Exception:
        return None


API_KEY = get_secret(
    "WATSONX_APIKEY"
)

PROJECT_ID = get_secret(
    "WATSONX_PROJECT_ID"
)

WATSONX_URL = get_secret(
    "WATSONX_URL"
)


if not API_KEY:
    raise ValueError(
        "WATSONX_APIKEY is missing from "
        "environment variables or Streamlit Secrets."
    )


if not PROJECT_ID:
    raise ValueError(
        "WATSONX_PROJECT_ID is missing from "
        "environment variables or Streamlit Secrets."
    )


if not WATSONX_URL:
    raise ValueError(
        "WATSONX_URL is missing from "
        "environment variables or Streamlit Secrets."
    )


class EcoSortLLM:

    def __init__(self):

        credentials = Credentials(
            url=WATSONX_URL,
            api_key=API_KEY
        )

        self.model = ModelInference(
            model_id="ibm/granite-4-h-small",
            credentials=credentials,
            project_id=PROJECT_ID
        )


    def answer(
        self,
        question,
        retrieved_documents
    ):

        context = "\n\n".join(
            [
                (
                    f"Source: {doc['filename']}, "
                    f"Page: {doc['page']}\n"
                    f"{doc['text']}"
                )
                for doc in retrieved_documents
            ]
        )


        prompt = f"""
You are EcoSort AI.

You provide sustainability and waste-management
information using the supplied source context.

IMPORTANT RULES:

1. Use the supplied context as the primary
   factual source.

2. Do not invent regulations.

3. Do not claim that a disposal method is officially
   required unless supported by the context.

4. If the context is insufficient, clearly say so.

5. Distinguish general guidance from local requirements.

6. Give a concise answer.

CONTEXT:

{context}

USER QUESTION:

{question}

Return:

Detected/mentioned item:
Waste category:
Recommended action:
Explanation:
Sources:
"""


        return self.model.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": 200
            }
        )
