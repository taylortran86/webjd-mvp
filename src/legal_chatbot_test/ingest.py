from .chroma_manager import add_documents_to_chroma_db, delete_chroma_db, query_chroma_db

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