"""
Système d'embedding flexible avec support multi-API et gestion des délais
"""
import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import hashlib
import requests
import os

from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    """Énumération des fournisseurs d'embedding supportés"""
    GEMINI = "gemini"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


@dataclass
class RetryConfig:
    """Configuration pour les tentatives de retry"""
    max_retries: int = 3
    base_delay: float = 1.0  # délai de base en secondes
    max_delay: float = 60.0  # délai maximum en secondes
    exponential_base: float = 2.0  # facteur pour backoff exponentiel
    jitter: bool = True  # ajouter du bruit pour éviter les collisions


@dataclass
class BatchConfig:
    """Configuration pour le traitement par batch"""
    batch_size: int = 10
    delay_between_batches: float = 1.0  # délai entre les batches en secondes
    max_concurrent_batches: int = 3


@dataclass
class EmbeddingResult:
    """Résultat d'une opération d'embedding"""
    embeddings: List[List[float]]
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0
    processing_time: float = 0.0
    tokens_used: Optional[int] = None


class BaseEmbeddingProvider(ABC):
    """Classe de base abstraite pour tous les fournisseurs d'embedding"""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None, 
                 batch_config: Optional[BatchConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.batch_config = batch_config or BatchConfig()
        self.last_request_time = 0.0
        self.request_count = 0
        
    @abstractmethod
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Méthode abstraite pour créer les embeddings"""
        pass
    
    @abstractmethod
    def get_rate_limit_delay(self) -> float:
        """Retourne le délai nécessaire avant la prochaine requête"""
        pass
    
    def _wait_for_rate_limit(self):
        """Attend le délai nécessaire pour respecter les limites de taux"""
        delay = self.get_rate_limit_delay()
        if delay > 0:
            logger.info(f"Attente de {delay:.2f}s pour respecter les limites de taux")
            time.sleep(delay)
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calcule le délai pour une tentative donnée"""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # jitter de ±25%
        
        return delay
    
    def embed_texts_with_retry(self, texts: List[str]) -> EmbeddingResult:
        """Embed des textes avec gestion des retries"""
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                self._wait_for_rate_limit()
                embeddings = self._embed_texts(texts)
                
                processing_time = time.time() - start_time
                self.last_request_time = time.time()
                self.request_count += 1
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    success=True,
                    retry_count=attempt,
                    processing_time=processing_time
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"Tentative {attempt + 1} échouée: {str(e)}")
                
                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(f"Nouvelle tentative dans {delay:.2f}s")
                    time.sleep(delay)
        
        processing_time = time.time() - start_time
        return EmbeddingResult(
            embeddings=[],
            success=False,
            error_message=str(last_error),
            retry_count=self.retry_config.max_retries,
            processing_time=processing_time
        )
    
    def embed_texts_batch(self, texts: List[str], 
                         progress_callback: Optional[Callable[[int, int], None]] = None) -> List[EmbeddingResult]:
        """Embed des textes par batch"""
        results = []
        total_batches = (len(texts) + self.batch_config.batch_size - 1) // self.batch_config.batch_size
        
        for i in range(0, len(texts), self.batch_config.batch_size):
            batch_num = i // self.batch_config.batch_size + 1
            batch_texts = texts[i:i + self.batch_config.batch_size]
            
            logger.info(f"Traitement du batch {batch_num}/{total_batches} ({len(batch_texts)} textes)")
            
            result = self.embed_texts_with_retry(batch_texts)
            results.append(result)
            
            if progress_callback:
                progress_callback(batch_num, total_batches)
            
            # Délai entre les batches (sauf pour le dernier)
            if i + self.batch_config.batch_size < len(texts):
                time.sleep(self.batch_config.delay_between_batches)
        
        return results


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Fournisseur d'embedding pour Google Gemini utilisant l'API HTTP directe"""
    
    def __init__(self, model_name: str = "models/text-embedding-004",
                 api_key: str = None,
                 retry_config: Optional[RetryConfig] = None,
                 batch_config: Optional[BatchConfig] = None):
        super().__init__(retry_config, batch_config)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Clé API Google requise. Définissez GOOGLE_API_KEY dans l'environnement ou passez api_key")
        
        # URL de l'API Gemini pour les embeddings
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:embedContent"
        
        # Configuration spécifique à Gemini
        if not batch_config:
            self.batch_config = BatchConfig(
                batch_size=20,  # Traitement plus conservateur par HTTP
                delay_between_batches=1.0,
                max_concurrent_batches=2
            )
        
        # Limites de taux pour Gemini (ajustez selon vos quotas)
        self.requests_per_minute = 60
        self.min_delay_between_requests = 1.0  # 1 seconde minimum
    
    def _embed_single_text(self, text: str) -> List[float]:
        """Embed un seul texte via l'API HTTP Gemini"""
        headers = {
            "Content-Type": "application/json"
        }
        
        params = {
            "key": self.api_key
        }
        
        data = {
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            params=params,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            error_msg = f"Erreur API Gemini: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        result = response.json()
        
        if "embedding" not in result:
            raise Exception(f"Réponse API inattendue: {result}")
        
        return result["embedding"]["values"]
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Implémentation spécifique pour Gemini avec appels HTTP directs"""
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self._embed_single_text(text)
                embeddings.append(embedding)
                
                # Petite pause entre les requêtes individuelles dans un batch
                if i < len(texts) - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'embedding du texte {i+1}/{len(texts)}: {str(e)}")
                raise
        
        return embeddings
    
    def get_rate_limit_delay(self) -> float:
        """Calcule le délai nécessaire pour respecter les limites Gemini"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_delay_between_requests:
            return self.min_delay_between_requests - time_since_last_request
        
        return 0.0
    
    def embed_query(self, query: str) -> List[float]:
        """Embed une seule requête"""
        return self._embed_single_text(query)


class ChromaCompatibleEmbedding:
    """Wrapper pour rendre notre système d'embedding compatible avec Chroma"""
    
    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        self.provider = embedding_provider
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Interface compatible Chroma pour embedder plusieurs documents"""
        result = self.provider.embed_texts_with_retry(texts)
        if not result.success:
            raise Exception(f"Échec de l'embedding: {result.error_message}")
        return result.embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Interface compatible Chroma pour embedder une requête"""
        if hasattr(self.provider, 'embed_query'):
            return self.provider.embed_query(text)
        else:
            # Fallback vers embed_texts pour les providers qui n'ont pas embed_query
            result = self.provider.embed_texts_with_retry([text])
            if not result.success:
                raise Exception(f"Échec de l'embedding: {result.error_message}")
            return result.embeddings[0]


class EmbeddingManager:
    """Gestionnaire principal pour les embeddings avec support multi-provider"""
    
    def __init__(self, provider: EmbeddingProvider = EmbeddingProvider.GEMINI,
                 retry_config: Optional[RetryConfig] = None,
                 batch_config: Optional[BatchConfig] = None):
        self.provider_type = provider
        self.retry_config = retry_config or RetryConfig()
        self.batch_config = batch_config or BatchConfig()
        self.provider = self._create_provider()
        self.cache = {}  # Cache simple pour éviter les requêtes dupliquées
    
    def _create_provider(self) -> BaseEmbeddingProvider:
        """Factory pour créer le bon provider"""
        if self.provider_type == EmbeddingProvider.GEMINI:
            return GeminiEmbeddingProvider(
                retry_config=self.retry_config,
                batch_config=self.batch_config
            )
        # Ajouter d'autres providers ici
        else:
            raise ValueError(f"Provider {self.provider_type} non supporté")
    
    def _get_text_hash(self, text: str) -> str:
        """Génère un hash pour mettre en cache"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_documents(self, documents: List[Document], 
                       use_cache: bool = True,
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """Embed une liste de documents avec leur métadonnées"""
        texts = [doc.page_content for doc in documents]
        
        # Vérifier le cache si activé
        if use_cache:
            cached_results = {}
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)
                if text_hash in self.cache:
                    cached_results[i] = self.cache[text_hash]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            logger.info(f"Cache: {len(cached_results)} hits, {len(uncached_texts)} nouveaux")
            texts_to_embed = uncached_texts
        else:
            texts_to_embed = texts
            uncached_indices = list(range(len(texts)))
        
        # Embed les textes non cachés
        results = []
        all_embeddings = [None] * len(documents)
        
        if texts_to_embed:
            batch_results = self.provider.embed_texts_batch(texts_to_embed, progress_callback)
            
            # Reconstituer les résultats complets
            embedding_idx = 0
            for result in batch_results:
                if result.success:
                    for embedding in result.embeddings:
                        original_idx = uncached_indices[embedding_idx]
                        all_embeddings[original_idx] = embedding
                        
                        # Mettre en cache si activé
                        if use_cache:
                            text_hash = self._get_text_hash(texts_to_embed[embedding_idx])
                            self.cache[text_hash] = embedding
                        
                        embedding_idx += 1
                
                results.append(result)
        
        # Ajouter les résultats cachés
        if use_cache:
            for i, embedding in cached_results.items():
                all_embeddings[i] = embedding
        
        # Statistiques
        successful_embeddings = sum(1 for emb in all_embeddings if emb is not None)
        total_processing_time = sum(r.processing_time for r in results)
        total_retries = sum(r.retry_count for r in results)
        
        return {
            "embeddings": all_embeddings,
            "documents": documents,
            "success_rate": successful_embeddings / len(documents),
            "total_processing_time": total_processing_time,
            "total_retries": total_retries,
            "batch_results": results,
            "cache_hits": len(cached_results) if use_cache else 0
        }
    
    def embed_chunks(self, chunks: List[str], 
                    progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """Embed une liste de chunks de texte"""
        # Créer des documents temporaires pour utiliser la même logique
        documents = [Document(page_content=chunk, metadata={"chunk_id": i}) 
                    for i, chunk in enumerate(chunks)]
        
        return self.embed_documents(documents, progress_callback=progress_callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du provider"""
        return {
            "provider": self.provider_type.value,
            "total_requests": self.provider.request_count,
            "cache_size": len(self.cache),
            "last_request_time": datetime.fromtimestamp(self.provider.last_request_time) if self.provider.last_request_time > 0 else None
        }
    
    def clear_cache(self):
        """Vide le cache"""
        self.cache.clear()
        logger.info("Cache vidé")
    
    def get_chroma_compatible_embedding(self) -> ChromaCompatibleEmbedding:
        """Retourne un objet compatible avec Chroma"""
        return ChromaCompatibleEmbedding(self.provider)


# Fonction utilitaire pour créer rapidement un manager
def create_embedding_manager(provider: str = "gemini",
                           max_retries: int = 3,
                           batch_size: int = 10,
                           delay_between_batches: float = 1.0) -> EmbeddingManager:
    """Factory function pour créer un EmbeddingManager avec des paramètres simples"""
    
    provider_enum = EmbeddingProvider(provider.lower())
    retry_config = RetryConfig(max_retries=max_retries)
    batch_config = BatchConfig(
        batch_size=batch_size,
        delay_between_batches=delay_between_batches
    )
    
    return EmbeddingManager(
        provider=provider_enum,
        retry_config=retry_config,
        batch_config=batch_config
    )


# Exemple d'usage
if __name__ == "__main__":
    # Exemple d'utilisation
    manager = create_embedding_manager(
        provider="gemini",
        max_retries=3,
        batch_size=5,
        delay_between_batches=2.0
    )
    
    # Textes d'exemple
    test_texts = [
        "Ceci est un premier texte à embedder",
        "Voici un second texte pour tester le système",
        "Un troisième texte pour compléter notre test"
    ]
    
    def progress_callback(batch_num, total_batches):
        print(f"Progression: {batch_num}/{total_batches} batches traités")
    
    # Test de l'embedding
    result = manager.embed_chunks(test_texts, progress_callback=progress_callback)
    
    print(f"Résultats:")
    print(f"- Taux de succès: {result['success_rate']:.2%}")
    print(f"- Temps de traitement: {result['total_processing_time']:.2f}s")
    print(f"- Nombre de retries: {result['total_retries']}")
    print(f"- Hits du cache: {result['cache_hits']}")
    
    # Statistiques
    stats = manager.get_stats()
    print(f"Statistiques: {stats}")
