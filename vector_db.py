from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os

class QdrantStorage:
    def __init__(self, url=os.getenv("QDRANT_PORT"), collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=60)
        self.collection = collection

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name = self.collection,
                vectors_config = VectorParams(size=dim, distance=Distance.COSINE)
            )
        
    def insert(self, ids, vectors, payloads, batch_size=20):
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_vectors = vectors[i:i+batch_size]
            batch_payloads = payloads[i:i+batch_size]

        points = [
            PointStruct(
                id=batch_ids[j],
                vector=batch_vectors[j],
                payload=batch_payloads[j]
            )
            for j in range(len(batch_ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def search(self, query_vector, top_k: int=5):
        results = self.client.query_points(
            collection_name = self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set()

        for res in results.points:
            payload = res.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

        if text:
            contexts.append(text)
            sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}