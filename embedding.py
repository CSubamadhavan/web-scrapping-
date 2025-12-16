import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


# -----------------------------------------------------------------------------
# Milvus helpers
# -----------------------------------------------------------------------------

def connect_milvus(host: str, port: int) -> None:
    """Connect to Milvus and register the default alias."""
    connections.connect(alias="default", host=host, port=port)


def ensure_collection(name: str, dim: int) -> Collection:
    """Return an existing collection or create a new one if missing."""
    if utility.has_collection(name):
        print(f"Collection '{name}' already exists. Using existing collection.")
        return Collection(name=name)  # Load existing collection
    
    # Create new collection only if it doesn't exist
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields, description="Web page content embeddings")
    collection = Collection(name=name, schema=schema)
    print(f"Created new collection '{name}' with 7 fields including user_id")
    return collection


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_rows(data: Dict[str, Any], model: SentenceTransformer) -> List[List[Any]]:
    ids, urls, titles, topics, contents, vectors, user_ids = [], [], [], [], [], [], []

    for idx, (_page, rec) in enumerate(data.items(), start=1):
        ids.append(idx)
        urls.append(rec.get("url", ""))
        titles.append(rec.get("title", ""))
        topics.append(rec.get("topic", ""))
        contents.append(rec.get("content", ""))
        vectors.append(model.encode(rec.get("content", ""), convert_to_numpy=True).tolist())
        user_ids.append(str(rec.get("user_id", "")))

    # Order matches schema: id, url, title, topic, content, vector, user_id
    return [ids, urls, titles, topics, contents, vectors, user_ids]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload content embeddings to Milvus")
    parser.add_argument("--json", default="D:/web/data/classified_data.json", help="Path to classified JSON")
    parser.add_argument("--model", default="multi-qa-MiniLM-L6-cos-v1", help="Sentence-Transformers model name")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--collection", default="webpages", help="Milvus collection name")
    args = parser.parse_args()

    # 1️⃣ Load model
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()

    # 2️⃣ Connect & create collection
    connect_milvus(args.host, args.port)
    collection = ensure_collection(args.collection, dim)

    # 3️⃣ Load data
    data = load_json(Path(args.json))
    rows = make_rows(data, model)

    # 4️⃣ Insert
    insert_result = collection.insert(rows)
    collection.flush()

    # 5️⃣ Index with COSINE metric
    index_params = {
        "metric_type": "COSINE",
        "index_type": "FLAT",
        "params": {}
    }
    collection.create_index(field_name="vector", index_params=index_params)

    print(f"✅ Inserted {len(rows[0])} rows into '{args.collection}'. IDs: {insert_result.primary_keys}")


if __name__ == "__main__":
    main()