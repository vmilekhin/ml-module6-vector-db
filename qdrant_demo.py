import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COL = "course_qdrant_demo"
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
client = QdrantClient(url=URL)

documents = [
    (1, "Кошка сидит на диване", 7),
    (2, "Кот спит дома", 7),
    (3, "Автомобиль движется по дороге", 8),
    (4, "Правила оформления отпуска", 7),
    (5, "Как оформить ежегодный отпуск", 8),
]

vectors = model.encode([x[1] for x in documents], normalize_embeddings=True)
dimension = int(vectors.shape[1])

if not client.collection_exists(COL):
    client.create_collection(
        COL,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE)
    )

client.upsert(
    collection_name=COL,
    wait=True,
    points=[
        models.PointStruct(
            id=doc_id,
            vector=vector.tolist(),
            payload={"text": content, "department_id": department}
        ) for (doc_id, content, department), vector in zip(documents, vectors)
    ]
)

query = "Как получить отпуск?"
query_vector = model.encode(query, normalize_embeddings=True).tolist()

for department in (None, 7):
    flt = None if department is None else models.Filter(
        must=[models.FieldCondition(key="department_id", match=models.MatchValue(value=department))]
    )
    result = client.query_points(
        collection_name=COL,
        query=query_vector,
        query_filter=flt,
        limit=3,
        with_payload=True
    )
    print(f"\nDepartment: {department}")
    for point in result.points:
        print(point.id, round(point.score, 4), point.payload)

print("\nГотово. Dashboard:", URL + "/dashboard")
