import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("SBERT_MODEL_NAME", "clip-ViT-B-32")

class statics:
    model = SentenceTransformer(MODEL_NAME)

if __name__ == "__main__":
    print("Model loaded successfully")