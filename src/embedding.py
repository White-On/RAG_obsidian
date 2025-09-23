from abc import ABC, abstractmethod
import requests
from dotenv import load_dotenv
import os
import ollama

load_dotenv()


@abstractmethod
class EmbeddingFactory(ABC):
    @abstractmethod
    def embed(self, text: str, **kwargs) -> list[float]:
        pass

    @staticmethod
    def create(embedding_method: str) -> "EmbeddingFactory":
        if embedding_method.lower() == "ollama":
            return EmbeddingOllama()
        elif embedding_method.lower() == "gemini":
            return EmbeddingGemini()
        else:
            raise ValueError(f"Modèle d'embedding inconnu: {embedding_method}")


class EmbeddingOllama(EmbeddingFactory):
    def __init__(self, model: str = "bge-m3"):
        self.client = ollama.Client()
        self.model = model

    def embed(self, text: str, **kwargs) -> list[float]:
        """Embed un seul texte via Ollama"""
        response = self.client.embeddings(model=self.model, 
                                          prompt=text, 
                                          **kwargs)
        return response.embedding

# options={
#         "temperature": 0.0,    # Contrôle de la randomness (0.0 = déterministe)
#         "top_k": 40,          # Top-k sampling
#         "top_p": 0.9,         # Top-p sampling
#         "repeat_penalty": 1.1, # Pénalité de répétition
#         "seed": 42,           # Seed pour la reproductibilité
#         "num_ctx": 2048,      # Taille du contexte
#         "num_predict": -1,    # Nombre de tokens à prédire (-1 = jusqu'à la fin)
#         "stop": ["<stop>"],   # Tokens d'arrêt
#     },

class EmbeddingGemini(EmbeddingFactory):
    API_URL = "https://generativelanguage.googleapis.com/v1beta2/models/textembedding-gecko-001:embed"
    API_KEY = os.getenv("GOOGLE_API_KEY")

    def embed(self, text: str) -> list[float]:
        """Embed un seul texte via l'API HTTP Gemini"""
        headers = {"Content-Type": "application/json"}

        params = {"key": self.API_KEY}

        data = {
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",  # QUESTION_ANSWERING
        }

        response = requests.post(
            self.API_URL, headers=headers, params=params, json=data, timeout=30
        )

        if response.status_code != 200:
            error_msg = f"Erreur API Gemini: {response.status_code} - {response.text}"
            print(error_msg)
            raise Exception(error_msg)

        result = response.json()

        if "embedding" not in result:
            raise Exception(f"Réponse API inattendue: {result}")

        return result["embedding"]["values"]
