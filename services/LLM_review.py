from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_paper_review(summaries, examples, paper_title: str = "") -> str:
    """
    Generate a detailed review of the research paper using summaries and example reviews.
    
    Args:
        summaries: ProcessedContent object containing text, table, and image summaries
        examples: Dict containing example papers and their reviews
        paper_title: Title of the paper being reviewed
    """
    
    
    text_sections = []
    for i, summary in enumerate(summaries.text_summaries, 1):
        if summary:
            text_sections.append(f"Section {i}: {summary}")
    text_content = "\n\n".join(text_sections)
    
    
    table_sections = []
    for i, summary in enumerate(summaries.table_summaries, 1):
        if summary:
            table_sections.append(f"Table {i}: {summary}")
    table_content = "\n\n".join(table_sections)
    
    
    image_sections = []
    for i, summary in enumerate(summaries.image_summaries, 1):
        if summary:
            image_sections.append(f"Figure {i}: {summary}")
    image_content = "\n\n".join(image_sections)
    
    
    doc_structure = f"""
    Document Analysis:
    - Text Sections: {len(summaries.text_summaries)} sections found
    - Tables: {len(summaries.table_summaries)} tables analyzed
    - Figures: {len(summaries.image_summaries)} figures processed
    """
    
    
    example_reviews = []
    if examples and "examples" in examples:
        for i, example in enumerate(examples["examples"], 1):
            for j, review in enumerate(example["reviews"], 1):
                example_reviews.append(
                    f"Example {i}.{j} (from paper: {example['title'][:100]}...):\n{review['comment']}"
                )
    
    
    prompt_template = """You are an expert academic reviewer analyzing a research paper in detail.
    Based on the provided content and example reviews, write a comprehensive academic review.

    Paper Title: {title}

    {doc_structure}

    === Content Analysis ===
    
    Text Content:
    {text_content}
    
    Table Analysis:
    {table_content}
    
    Figures Analysis:
    {image_content}
    
    === Review Guidelines ===
    
    Example Review Styles:
    {example_styles}
    
    Review Structure Requirements:
    1. Overview and Contribution (200-250 words)
       - Paper's main objectives
       - Key contributions to the field
       - Context and relevance
    
    2. Methodology Analysis (200-250 words)
       - Research approach
       - Technical implementation
       - Use of neural networks and image processing
    
    3. Results and Implementation (200-250 words)
       - Key findings
       - Technical achievements
       - Performance metrics
    
    4. Critical Analysis (150-200 words)
       - Strengths and limitations
       - Areas for improvement
       - Future research directions
    
    5. Conclusion (100-150 words)
       - Overall assessment
       - Recommendation
       - Impact on the field
    
    Additional Guidelines:
    - Reference specific sections, tables, and figures
    - Maintain academic tone while being engaging
    - Provide constructive criticism
    - Compare with state-of-the-art approaches
    - Discuss practical applications

    Write your detailed review following this structure:
    """
    
    model = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.65,
        max_tokens=2048
    )
    

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | StrOutputParser()
    
    try:

        example_text = "\n\n".join([
            rev[:500] + "..." for rev in example_reviews[:3]
        ])
        
        
        review = chain.invoke({
            "title": paper_title,
            "doc_structure": doc_structure,
            "text_content": text_content,
            "table_content": table_content,
            "image_content": image_content,
            "example_styles": example_text
        })
        
        formatted_review = f"""# Review of {paper_title}

{review}

---
*Review generated using advanced language model analysis*
"""
        return formatted_review
        
    except Exception as e:
        print(f"Error generating review: {str(e)}")
        return "Error: Could not generate review"