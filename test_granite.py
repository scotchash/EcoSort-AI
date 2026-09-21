import os

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference


# Load environment variables
load_dotenv()

API_KEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")


# Validate environment variables
if not API_KEY:
    raise ValueError("WATSONX_APIKEY is missing from .env")

if not PROJECT_ID:
    raise ValueError("WATSONX_PROJECT_ID is missing from .env")

if not WATSONX_URL:
    raise ValueError("WATSONX_URL is missing from .env")


# IBM watsonx credentials
credentials = Credentials(
    url=WATSONX_URL,
    api_key=API_KEY
)


# Granite model
model = ModelInference(
    model_id="ibm/granite-4-h-small",
    credentials=credentials,
    project_id=PROJECT_ID
)


# EcoSort AI prompt
prompt = """
You are EcoSort AI, an AI assistant for sustainable waste management.

Explain in 3-4 clear sentences why proper waste segregation is important.
"""


# Generate response
response = model.generate_text(
    prompt=prompt,
    params={
        "max_new_tokens": 100
    }
)


print("\n===== EcoSort AI =====\n")
print(response)