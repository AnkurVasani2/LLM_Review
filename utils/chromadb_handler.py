import uuid
from dataclasses import asdict
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

class ChromaDBHandler:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            collection_name="multi_modal_rag",
            embedding_function=self.embedding_function
        )
        self.store = InMemoryStore()
        
    def store_content(self, content):
        """Store processed content in ChromaDB"""
        retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key="doc_id"
        )
        
        for content_type in ['text', 'table', 'image']:
            self._store_type_content(
                retriever,
                getattr(content, f"{content_type}_summaries"),
                getattr(content, f"{content_type}s"),
                content_type
            )
            
        return retriever, self.vectorstore
        
    def _store_type_content(self, retriever, summaries, originals, doc_type):
        """Store specific content type in ChromaDB"""
        if not summaries or not originals:
            print(f"No {doc_type} content to store")
            return
            
        valid_pairs = [
            (summary, original) 
            for summary, original in zip(summaries, originals)
            if summary is not None and original is not None 
            and str(summary).strip() and str(original).strip()
        ]
        
        if not valid_pairs:
            print(f"No valid {doc_type} pairs to store")
            return
            
        summaries, originals = zip(*valid_pairs)
        ids = [str(uuid.uuid4()) for _ in originals]
        
        try:
            summary_docs = [
                Document(
                    page_content=str(summary), 
                    metadata={"doc_id": id_, "type": f"{doc_type}_summary"}
                )
                for id_, summary in zip(ids, summaries)
            ]
            
            original_docs = [
                Document(
                    page_content=str(original),
                    metadata={"doc_id": id_, "type": f"{doc_type}_original"}
                )
                for id_, original in zip(ids, originals)
            ]
            
            if summary_docs:
                retriever.vectorstore.add_documents(summary_docs)
            if original_docs:
                retriever.vectorstore.add_documents(original_docs)
            retriever.docstore.mset(list(zip(ids, originals)))
            
            print(f"Stored {len(ids)} {doc_type} pairs")
            
        except Exception as e:
            print(f"Error storing {doc_type} content: {str(e)}")
            
    def display_stored_content(self):
        """Display all stored content in ChromaDB"""
        print("\n=== Stored Content in ChromaDB ===")
        
        # Check content types
        for content_type in ['text', 'table', 'image']:
            print(f"\n--- {content_type.title()} Content ---")
            
            # Get summaries
            summaries = self.vectorstore.similarity_search(
                content_type,
                filter={"type": f"{content_type}_summary"},
                k=100
            )
            print(f"\nFound {len(summaries)} {content_type} summaries:")
            for i, doc in enumerate(summaries[:3], 1):  # Show first 3
                print(f"\n{i}. Summary (ID: {doc.metadata['doc_id']}):")
                print(f"   {doc.page_content[:200]}...")
                
            # Get originals
            originals = self.vectorstore.similarity_search(
                content_type,
                filter={"type": f"{content_type}_original"},
                k=100
            )
            print(f"\nFound {len(originals)} {content_type} originals:")
            for i, doc in enumerate(originals[:3], 1):  # Show first 3
                print(f"\n{i}. Original (ID: {doc.metadata['doc_id']}):")
                print(f"   {doc.page_content[:200]}...")