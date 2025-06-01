import uuid
from langchain_chroma import Chroma  # Updated import
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.retrievers.multi_vector import MultiVectorRetriever

def store_in_chromadb(text_summaries, texts, table_summaries, tables, image_summaries, images):
    """
    Stores summaries and their corresponding original data in ChromaDB,
    linking them via a shared doc_id.
    """
    
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=embedding_function
    )
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    text_pairs = [
        (summary, text)
        for summary, text in zip(text_summaries, texts)
        if summary and str(summary).strip() and text and str(text).strip()
    ]
    table_pairs = [
        (summary, table)
        for summary, table in zip(table_summaries, tables)
        if summary and str(summary).strip() and table and str(table).strip()
    ]
    image_pairs = [
        (summary, image)
        for summary, image in zip(image_summaries, images)
        if summary and str(summary).strip() and image and str(image).strip()
    ]

    # Unzip pairs back to lists
    text_summaries, texts = zip(*text_pairs) if text_pairs else ([], [])
    table_summaries, tables = zip(*table_pairs) if table_pairs else ([], [])
    image_summaries, images = zip(*image_pairs) if image_pairs else ([], [])

    # Convert tuples to lists
    text_summaries, texts = list(text_summaries), list(texts)
    table_summaries, tables = list(table_summaries), list(tables)
    image_summaries, images = list(image_summaries), list(images)

    # Store text summaries and originals
    if text_summaries and texts:
        text_ids = [str(uuid.uuid4()) for _ in texts]
        summary_text_docs = [
            Document(page_content=summary, metadata={id_key: text_ids[i], "type": "text_summary"})
            for i, summary in enumerate(text_summaries)
        ]
        original_text_docs = [
            Document(page_content=texts[i], metadata={id_key: text_ids[i], "type": "text_original"})
            for i in range(len(texts))
        ]
        if summary_text_docs:
            retriever.vectorstore.add_documents(summary_text_docs)
        if original_text_docs:
            retriever.vectorstore.add_documents(original_text_docs)
        retriever.docstore.mset(list(zip(text_ids, texts)))

    if table_summaries and tables:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_table_docs = [
            Document(page_content=summary, metadata={id_key: table_ids[i], "type": "table_summary"})
            for i, summary in enumerate(table_summaries)
        ]
        original_table_docs = [
            Document(page_content=tables[i], metadata={id_key: table_ids[i], "type": "table_original"})
            for i in range(len(tables))
        ]
        if summary_table_docs:
            retriever.vectorstore.add_documents(summary_table_docs)
        if original_table_docs:
            retriever.vectorstore.add_documents(original_table_docs)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    if image_summaries and images:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img_docs = [
            Document(page_content=summary, metadata={id_key: img_ids[i], "type": "image_summary"})
            for i, summary in enumerate(image_summaries)
        ]
        original_img_docs = [
            Document(page_content=images[i], metadata={id_key: img_ids[i], "type": "image_original"})
            for i in range(len(images))
        ]
        if summary_img_docs:
            retriever.vectorstore.add_documents(summary_img_docs)
        if original_img_docs:
            retriever.vectorstore.add_documents(original_img_docs)
        retriever.docstore.mset(list(zip(img_ids, images)))

    print("All summaries and originals stored in ChromaDB with doc_id links.")
    return retriever, vectorstore, store