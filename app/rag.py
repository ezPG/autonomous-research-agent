import faiss
import numpy as np

class RAGStore:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            # Suppress logs if possible (some environments print to stdout)
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model

    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.text_chunks = []
        self.metadata = [] 

    def chunk_text(self, text, chunk_size=500):
        words = text.split()
        for i in range(0, len(words), chunk_size):
            yield " ".join(words[i:i + chunk_size])

    def add_document(self, text: str, source: str = ""):
        chunks = list(self.chunk_text(text))
        if not chunks:
            return
        embeddings = self.get_model().encode(chunks)
        self.index.add(np.array(embeddings).astype("float32"))
        self.text_chunks.extend(chunks)
        self.metadata.extend([{"source": source}] * len(chunks))

    def retrieve(self, query: str, k=3):
        if not self.text_chunks:
            return []
            
        q_emb = self.get_model().encode([query])
        distances, indices = self.index.search(
            np.array(q_emb).astype("float32"), k
        )
        results = []
        for i in indices[0]:
            if i >= 0 and i < len(self.text_chunks):
                results.append({
                    "text": self.text_chunks[i],
                    "metadata": self.metadata[i]
                })
        return results
