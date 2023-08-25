# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from typing import List
from cog import BasePredictor, Input, BaseModel
from sentence_transformers import SentenceTransformer


class Output(BaseModel):
    vectors: List[float]
    text: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory"""
        self.model = SentenceTransformer('thenlper/gte-base')

    def predict(
        self,
        text: str = Input(description="Text string to embed"),
    ) -> Output:
        """Run a single prediction on the model"""

        # Embed the text
        embeddings = self.model.encode([text])

        return Output(vectors=embeddings[0].tolist(), text=text)
