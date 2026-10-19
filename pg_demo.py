import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = os.getenv("PG_DSN", "dbname=postgres user=postgres password=postgres host=127.0.0.1")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Наши тестовые документы (те же, что были в Qdrant)
DOCS = [
    (1, "Кошка сидит на диване", 7),
    (2, "Кот спит дома", 7),
    (3, "Автомобиль движется по дороге", 8),
    (4, "Правила оформления отпуска", 7),
    (5, "Как оформить ежегодный отпуск", 8),
]

def get_embeddings(texts):
    model = SentenceTransformer(MODEL)
    return model.encode(texts, normalize_embeddings=True)

def pg_init():
    print("Инициализация таблицы и расширения pgvector...")
    with psycopg.connect(DSN) as conn:
        # Включаем расширение
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Создаем таблицу под 384 компонента (MiniLM)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS module6_documents (
                id bigint PRIMARY KEY,
                content text NOT NULL,
                department_id integer NOT NULL,
                model_name text NOT NULL,
                embedding vector(384) NOT NULL
            );
        """)
    print("Таблица готова.")

def pg_insert():
    print("Считаем эмбеддинги и вставляем в БД...")
    vectors = get_embeddings([d[1] for d in DOCS])
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        for (id_, txt, dep), v in zip(DOCS, vectors):
            conn.execute("""
                INSERT INTO module6_documents (id, content, department_id, model_name, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    department_id = EXCLUDED.department_id,
                    embedding = EXCLUDED.embedding;
            """, (id_, txt, dep, MODEL, v))
    print(f"Успешно вставлено {len(DOCS)} строк.")

def pg_query(text, department=None):
    print(f"\nИщем: '{text}' (Фильтр отдела: {department})")
    q_vec = get_embeddings([text])[0]
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        # Оператор <=> — это косинусное расстояние в pgvector
        rows = conn.execute("""
            SELECT id, content, department_id, embedding <=> %s AS distance
            FROM module6_documents
            WHERE model_name = %s
              AND (%s::integer IS NULL OR department_id = %s)
            ORDER BY embedding <=> %s
            LIMIT 3;
        """, (q_vec, MODEL, department, department, q_vec)).fetchall()
        
        for row in rows:
            print(f"ID: {row[0]}, Расстояние: {row[3]:.4f}, Текст: '{row[1]}', Отдел: {row[2]}")

if __name__ == "__main__":
    pg_init()
    pg_insert()
    pg_query("Как получить отпуск?")          # Без фильтра
    pg_query("Как получить отпуск?", 7)       # Только 7-й отдел
