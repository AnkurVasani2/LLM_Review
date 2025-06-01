from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
from typing import Optional


from services import process_pdf, generate_paper_review
from utils import ChromaDBHandler, KeywordExtractor, get_examples_from_keywords


class ReviewResponse(BaseModel):
    paper_title: str
    keyword_count: int
    keywords: list
    example_count: int
    review: str
    status: str

# Initialize handlers
db_handler = ChromaDBHandler()
keyword_extractor = KeywordExtractor()

router = APIRouter(
    prefix="/paper",
    tags=["paper review"]
)

@router.post("/analyze", response_model=ReviewResponse)
async def analyze_paper(
    file: UploadFile = File(...),
    paper_title: Optional[str] = None
) -> JSONResponse:
    """
    Analyze a research paper and generate a detailed review.
    
    Args:
        file: PDF file of the research paper
        paper_title: Optional title of the paper
    
    Returns:
        JSONResponse containing the analysis results
    """
    try:
        
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if not paper_title:
            paper_title = os.path.splitext(file.filename)[0]

        summaries = process_pdf(file_path)
        
        retriever, vectorstore = db_handler.store_content(summaries)
        
        keywords = keyword_extractor.extract_from_vectorstore(retriever, vectorstore)
        
        if keywords and "keywords" in keywords:
            examples = get_examples_from_keywords(keywords)
        else:
            examples = {"total_examples": 0, "examples": []}
            
        review = generate_paper_review(
            summaries=summaries,
            examples=examples,
            paper_title=paper_title
        )
        
        response_data = {
            "paper_title": paper_title,
            "keyword_count": len(keywords.get("keywords", [])),
            "keywords": keywords.get("keywords", []),
            "example_count": examples.get("total_examples", 0),
            "review": review,
            "status": "success"
        }
        
        os.remove(file_path)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing paper: {str(e)}"
        )