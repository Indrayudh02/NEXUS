import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


load_dotenv()


class R(BaseModel):
    question: str


api_key = os.getenv("GEMINI_FALLBACK_KEY")
model = os.getenv("GEMINI_FALLBACK_MODEL")

print("MODEL:", model)
print("KEY:", "FOUND" if api_key else "MISSING")


client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model=model,
    contents=(
        "Generate one focused research question for this corpus gap. "
        "Return JSON with one field called question."
    ),
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=R,
        temperature=0.1,
    ),
)

print("RESPONSE:")
print(response.text)