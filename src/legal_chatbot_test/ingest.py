from dotenv import load_dotenv
import os
import shutil
from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from pathlib import Path
import json
from hashlib import sha256

load_dotenv()

embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

curr_file = Path(__file__).resolve()
project_root = curr_file.parents[2]
relative_db_path = os.getenv("DATABASE_LOCATION")

CHROMA_DB_PATH = str(project_root.joinpath(relative_db_path))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


# ----- Chroma DB Functions -----
def get_chroma_db() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

def delete_chroma_db() -> None:
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Deleting Chroma DB at {CHROMA_DB_PATH}...")
        shutil.rmtree(CHROMA_DB_PATH)
        print("Chroma DB deleted.")
    else:
        print("No Chroma DB found to delete.")


def add_documents_to_chroma_db() -> None:
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        print(f"[add_documents_to_chroma_db] Data directory not found at: {data_dir}")
        return

    vectorstore = get_chroma_db()

    # Load existing chunks and their metadata & ids from the DB
    print("[add_documents_to_chroma_db] Loading existing documents from Chroma DB...")
    existing_docs = vectorstore.get()
    existing_metadatas = existing_docs["metadatas"]
    existing_ids = existing_docs["ids"]

    # Group existing chunk IDs by 'source'
    source_to_existing_ids = {}
    source_to_existing_hashes = {}

    for meta, _id in zip(existing_metadatas, existing_ids):
        source = meta.get("source")
        hash_val = meta.get("hash")
        if source:
            source_to_existing_ids.setdefault(source, []).append(_id)
            source_to_existing_hashes.setdefault(source, set()).add(hash_val)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    all_new_chunks: List[Document] = []
    sources_to_delete = set()

    for file_path in data_dir.glob("*"):
        if file_path.is_file():
            print(f"[add_documents_to_chroma_db] Processing {file_path.name}...")

            # Load documents from file
            docs = load_documents_from_file(str(file_path))
            print(f"  Loaded {len(docs)} full documents.")

            # For each doc, decide if it's new or updated (based on source + hash)
            for doc in docs:
                source = doc.metadata.get("source")
                new_hash = doc.metadata.get("hash")

                # If source exists and hash differs → mark old chunks for deletion
                if source in source_to_existing_hashes:
                    if new_hash not in source_to_existing_hashes[source]:
                        print(f"  Document with source {source} updated - marking old chunks for deletion.")
                        sources_to_delete.add(source)
                else:
                    # New source, no need to delete
                    pass

            # Now delete old chunks for all sources marked
            for source in sources_to_delete:
                ids_to_delete = source_to_existing_ids.get(source, [])
                if ids_to_delete:
                    print(f"  Deleting {len(ids_to_delete)} old chunks for source: {source}")
                    vectorstore.delete(ids=ids_to_delete)
                    # Remove deleted sources from existing maps to avoid duplication
                    source_to_existing_ids.pop(source, None)
                    source_to_existing_hashes.pop(source, None)

            # Split documents into chunks, add chunk_index metadata
            chunks = text_splitter.split_documents(docs)

            hash_to_index = {}
            updated_chunks = []
            for chunk in chunks:
                doc_hash = chunk.metadata.get("hash")
                if doc_hash not in hash_to_index:
                    hash_to_index[doc_hash] = 0
                chunk_index = hash_to_index[doc_hash]

                new_metadata = dict(chunk.metadata)
                new_metadata["chunk_index"] = chunk_index
                new_chunk = Document(page_content=chunk.page_content, metadata=new_metadata)
                updated_chunks.append(new_chunk)

                hash_to_index[doc_hash] += 1

            # Filter out chunks that already exist (based on hash + source)
            fresh_chunks = []
            for chunk in updated_chunks:
                source = chunk.metadata.get("source")
                doc_hash = chunk.metadata.get("hash")

                # If source still exists in DB and hash matches, chunk already exists → skip
                if source in source_to_existing_hashes and doc_hash in source_to_existing_hashes[source]:
                    continue
                fresh_chunks.append(chunk)

            print(f"  Adding {len(fresh_chunks)} new chunks to vector store.")
            all_new_chunks.extend(fresh_chunks)

    if not all_new_chunks:
        print("[add_documents_to_chroma_db] No new or updated chunks to add.")
        return

    vectorstore.add_documents(all_new_chunks)
    print(f"[add_documents_to_chroma_db] Added {len(all_new_chunks)} new/updated chunks to Chroma DB.")


def query_chroma_db(query: str, k: int = 5, score_threshold: float = 0.5):
    vectorstore = get_chroma_db()

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )

    results = retriever.invoke(query)

    # if not results:
    #     print("No relevant documents found.")
    #     return []

    print(f"{len(results)} results for query: '{query}':\n")
    # for i, doc in enumerate(results[:k], start=1):
    #     print(f"--- Document {i} ---")
    #     print(f"Content: {doc.page_content}")  # Full chunk text
    #     print(f"Metadata: {doc.metadata}")
    #     print()
    # print(results)
    return results


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


# ----- Main Function -----

if __name__ == "__main__":
    print("Clearing old DB...")
    delete_chroma_db()

    print("Adding documents to Chroma DB...")
    add_documents_to_chroma_db()

    test_query = "divorce definition"
    print(f"Querying: {test_query}")
    query_chroma_db(test_query)
    
    # test_query = "How do I get an annulment?"
    # print(f"Querying: {test_query}")
    # query_chroma_db(test_query)
