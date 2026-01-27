import faiss
import numpy as np

class RAGStore:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model

    def __init__(self, persist_dir="data/rag_index"):
        self.persist_dir = persist_dir
        self.index = faiss.IndexFlatL2(384)
        self.text_chunks = []
        self.metadata = [] 
        self.load()

    def save(self):
        import os
        import pickle
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
        
        faiss.write_index(self.index, os.path.join(self.persist_dir, "index.faiss"))
        with open(os.path.join(self.persist_dir, "data.pkl"), "wb") as f:
            pickle.dump({"chunks": self.text_chunks, "metadata": self.metadata}, f)

    def load(self):
        import os
        import pickle
        index_path = os.path.join(self.persist_dir, "index.faiss")
        data_path = os.path.join(self.persist_dir, "data.pkl")
        
        if os.path.exists(index_path) and os.path.exists(data_path):
            self.index = faiss.read_index(index_path)
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                self.text_chunks = data["chunks"]
                self.metadata = data["metadata"]

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
        self.save() # Auto-persist after adding

    def clear(self):
        import os
        import shutil
        self.index = faiss.IndexFlatL2(384)
        self.text_chunks = []
        self.metadata = []
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

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
