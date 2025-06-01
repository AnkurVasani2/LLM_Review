import os
from dotenv import load_dotenv
from services.integrated_retriver import process_pdf
from services.LLM_review import generate_paper_review
from utils.chromadb_handler import ChromaDBHandler
from utils.keyword_extractor import KeywordExtractor
from utils.get_examples import get_examples_from_keywords

def main():
    """Main entry point for PDF processing pipeline"""
    load_dotenv()
    print("Starting PDF processing pipeline...")
    
    pdf_path = r"F:\IITP_Research_Task\LLM_review\dataset\772.pdf"
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Step 1: Process PDF and get summaries
        summaries = process_pdf(pdf_path)
        
        # Step 2: Store in ChromaDB
        db_handler = ChromaDBHandler()
        retriever, vectorstore = db_handler.store_content(summaries)
        
        # Display stored content
        db_handler.display_stored_content()
        
        # Step 3: Extract and display keywords
        keyword_extractor = KeywordExtractor()
        keywords = keyword_extractor.extract_from_vectorstore(retriever, vectorstore)
        
        
        if keywords and "keywords" in keywords:
            print(f"\nFound {len(keywords['keywords'])} relevant keywords")
        
        # Step 4: Get examples for keywords
        if keywords and "keywords" in keywords:
            print("\nFinding relevant examples from dataset...")
            examples = get_examples_from_keywords(keywords)
            
            # Print examples
            print(f"\n=== Found {examples['total_examples']} Example Papers ===")
            for i, example in enumerate(examples['examples'], 1):
                print(f"\nPaper {i}:")
                print(f"Title: {example['title']}")
                print(f"Matching keywords: {', '.join(example['matching_keywords'])}")
                print(f"Abstract excerpt: {example['abstract'][:200]}...")
                print(f"Reviews ({len(example['reviews'])}):") 
                for j, review in enumerate(example['reviews'], 1):
                    print(f"  Review {j}: {review['comment'][:100]}...")

        # Step 5: Generate detailed review
        print("\nGenerating detailed paper review...")
        review = generate_paper_review(
            summaries=summaries,
            examples=examples,
            paper_title="Neural Networks for Instance-Level Image Retrieval"  # Add paper title
        )
        
        
        print("\n=== Generated Paper Review ===")
        print(review)
        
        print("\nProcessing completed successfully!")
        return summaries, keywords, examples, review
        
    except Exception as e:
        print(f"Error in processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()

