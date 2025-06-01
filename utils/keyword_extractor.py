from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.get_examples import get_examples_from_keywords

class KeywordExtractor:
    def __init__(self, model_name="meta-llama/llama-4-maverick-17b-128e-instruct"):
        self.model = ChatGroq(temperature=0.3, model=model_name)
        self.parser = JsonOutputParser()
        
    def extract_from_vectorstore(self, retriever, vectorstore):
        """Extract keywords from stored content"""
        
        summaries = self._get_all_summaries(vectorstore)
        if not summaries:
            return {"keywords": []}
            
        
        keywords = self._extract_keywords(summaries)
        
        
        print("\n=== Extracted Keywords ===")
        if isinstance(keywords, dict) and "keywords" in keywords:
            for kw in keywords["keywords"]:
                print(f"- {kw['term']:<30} (importance: {kw['importance']:.2f})")
        
        return keywords
        
    def _get_all_summaries(self, vectorstore):
        """Get all summaries from vectorstore"""
        results = []
        for doc_type in ['text', 'table', 'image']:
            docs = vectorstore.similarity_search(
                doc_type,
                filter={"type": f"{doc_type}_summary"},
                k=100
            )
            results.extend([doc.page_content for doc in docs])
        return results
        
    def _extract_keywords(self, summaries):
        """Extract keywords using LLM"""
        
        prompt_template = """You are an expert at extracting key technical concepts from academic text.
        Analyze these summaries and extract the most relevant keywords and their importance.
        
        Requirements:
        1. Focus on technical terms, methodologies, and key concepts
        2. Include multi-word phrases where appropriate
        3. Rate importance from 0.0 to 1.0 (1.0 being most important)
        4. Return in valid JSON format
        5. Limit to 15 most important keywords
        
        Summaries to analyze:
        {text}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.model | self.parser
        
        try:
            
            combined_text = "\n\n".join(summaries)
            response = chain.invoke({"text": combined_text})
            
            # Format results for better readability
            if isinstance(response, dict) and "keywords" in response:
                print("\n=== Extracted Keywords ===")
                for kw in response["keywords"]:
                    print(f"- {kw['term']:<30} (importance: {kw['importance']:.2f})")
            return response
            
        except Exception as e:
            print(f"Error extracting keywords: {str(e)}")
            return {"keywords": []}