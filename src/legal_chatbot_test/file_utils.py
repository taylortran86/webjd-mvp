import json
from hashlib import sha256
from typing import List
from langchain_core.documents import Document

# ----- File Functions -----

def compute_hash_from_string(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()

def load_documents_from_file(file_path: str) -> List[Document]:
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                content = obj.get("content", "").strip()
                if not content:
                    continue  # skip empty docs

                content_hash = compute_hash_from_string(content)

                metadata = {
                    "source": obj.get("source", "").strip(),
                    "title": obj.get("title", "").strip(),
                    "hash": content_hash,
                }
                documents.append(Document(page_content=content, metadata=metadata))
            except json.JSONDecodeError as e:
                print(f"[load_documents_from_file] Skipping line {line_num} in {file_path}: Invalid JSON - {e}")
    return documents
