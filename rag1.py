from sentence_transformers import SentenceTransformer
import chromadb
from knowledge import documents

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="music_docs")

embeddings = embedder.encode(documents).tolist()
collection.upsert(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc{i}" for i in range(len(documents))]
)

def _query(question, n_results=2):
    question_embedding = embedder.encode([question]).tolist()
    return collection.query(query_embeddings=question_embedding, n_results=n_results)

def retrieve(question, n_results=2):
    results = _query(question, n_results)
    return results["documents"][0]


if __name__ == "__main__":
    for q in ["How long is a quarter note?", "What instrument did Adolphe Sax invent?", "What is the capital of Japan?"]:
        results1 = _query(q)
        for doc, dist in zip(results1["documents"][0], results1["distances"][0]):
            print(f"{dist:.3f} - {doc}")
        print()

