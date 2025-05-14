from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"hello": "abodient"}

# ----------  classifier endpoint ----------
from pydantic import BaseModel
from agents.classifier import classify
from agents.clarifier import clarify


class TextItem(BaseModel):
    text: str

@app.post("/classify")
async def classify_ep(item: TextItem):
    """
    Given tenant text, return urgency & responsibility.
    """
    return classify(item.text)

@app.post("/clarify")
def clarify_ep(item: TextItem):
    return clarify(item.text)


