# based upon the extracted keyword get example reviews from the dataset

import os
import json
import csv
from typing import List, Dict

def get_examples_from_keywords(keywords_dict: Dict) -> Dict:
    """
    Get top 5 example papers total across all keywords, with up to 2 reviews each.
    
    Args:
        keywords_dict (dict): Dictionary containing keywords from KeywordExtractor
        
    Returns:
        dict: JSON formatted dictionary with examples and their matching keywords
    """
    
    keywords = [
        item["term"] 
        for item in sorted(
            keywords_dict["keywords"], 
            key=lambda x: x["importance"], 
            reverse=True
        )
    ]
    
    all_examples = []
    
    file_path = r"F:\IITP_Research_Task\LLM_review\output\reviews.csv"
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            
            review_pairs = []
            for i in range(1, 36):
                title_col = f'Review_{i}_Title'
                comment_col = f'Review_{i}_Comments'
                if title_col in headers and comment_col in headers:
                    review_pairs.append((title_col, comment_col))
            
            
            for row in reader:
                if len(all_examples) >= 5:  # Stop after finding 5 examples
                    break
                    
                abstract = row.get('Abstract', '').strip()
                title = row.get('Title', '').strip()
                
                if not abstract or not title:
                    continue
                
                
                matching_keywords = [
                    kw for kw in keywords 
                    if kw.lower() in abstract.lower()
                ]
                
                if matching_keywords:  
                    reviews = []
                    for title_col, comment_col in review_pairs:
                        if len(reviews) >= 2:
                            break
                        
                        review_title = row.get(title_col, '').strip()
                        review_comment = row.get(comment_col, '').strip()
                        
                        if review_title and review_comment:
                            reviews.append({
                                "title": review_title,
                                "comment": review_comment
                            })
                    
                    
                    if reviews:
                        all_examples.append({
                            "title": title,
                            "abstract": abstract,
                            "reviews": reviews,
                            "matching_keywords": matching_keywords
                        })
    
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return {"examples": []}
    
    result = {
        "total_examples": len(all_examples),
        "examples": all_examples
    }
    
    return result
