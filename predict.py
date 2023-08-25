# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from typing import List, TypedDict
from cog import BasePredictor, Input
from sentence_transformers import SentenceTransformer


class Embedding(TypedDict):
    vectors: List[float]
    text: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory"""
        self.model = SentenceTransformer('thenlper/gte-base')

    def predict(
        self,
        text: str = Input(description="Text string to embed"),
    ) -> Embedding:
        """Run a single prediction on the model"""

        # Encode the chunks
        result = self.model.encode([text])

        return Embedding(vectors=result[0].tolist(), text=text)
