import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

def get_text_and_tables(chunks):
    """Separate text and table elements from chunks."""
    text_chunks = []
    table_chunks = []
    for chunk in chunks:
        type_str = str(type(chunk))
        if "Table" in type_str:
            table_chunks.append(getattr(chunk.metadata, "text_as_html", str(chunk)))
        elif "CompositeElement" in type_str:
            text_chunks.append(str(chunk))
    return text_chunks, table_chunks

def summarize_texts(texts, model_temp=0.5, model_name="meta-llama/llama-4-maverick-17b-128e-instruct"):
    print(f"Summarizing {len(texts)} text elements with model: {model_name} at temperature: {model_temp}")
    prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Table or text chunk: {element}
        
"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=model_temp, model=model_name)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    return summarize_chain.batch(texts, {"max_concurrency": 1})

def summarize_images(images_base64=None, text_summary=None):
    """Summarize images with optional text context."""

    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    if not os.path.exists(figures_dir):
        print(f"Figures directory not found: {figures_dir}")
        return [], []

    valid_images = []
    for img_file in os.listdir(figures_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(figures_dir, img_file)
            try:
                with open(img_path, 'rb') as f:
                    import base64
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    img_data = f"data:image/jpeg;base64,{img_data}"
                    valid_images.append(img_data)
                    print(f"Valid image found: {img_file}")
            except Exception as e:
                print(f"Error reading image {img_file}: {str(e)}")

    if not valid_images:
        print("No valid images found to summarize")
        return [], []

    context = f"Summary of the research paper: {text_summary}" if text_summary else ""
    prompt_template = f"""Analyze this research paper image in detail.
Context: {context}
Focus on:
1. Type of visualization (graph, diagram, architecture, etc.)
2. Key elements and their relationships
3. Main findings or patterns shown
4. Technical details if present"""

    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": {"url": "{image}"}}
            ],
        )
    ]
    
    prompt = ChatPromptTemplate.from_messages(messages)
    model = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.3)
    chain = prompt | model | StrOutputParser()
    
    try:
        summaries = []
        for img in valid_images:
            try:
                response = chain.invoke({"image": img})
                summaries.append(response)
                print(f"Generated summary for image")
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                summaries.append(None) 
        
        valid_pairs = [(s, img) for s, img in zip(summaries, valid_images) if s is not None]
        if valid_pairs:
            summaries, images = zip(*valid_pairs)
            return list(summaries), list(images)
        return [], []
        
    except Exception as e:
        print(f"Error in image summarization: {str(e)}")
        return [], []

def summarize_chunks(chunks):
    print(f"Summarizing {len(chunks)} chunks")
    text_chunks, table_chunks = get_text_and_tables(chunks)
    text_summaries = summarize_texts(text_chunks)
    table_summaries = summarize_texts(table_chunks)
    text_summaries = [s for s in text_summaries if s and str(s).strip()]
    table_summaries = [s for s in table_summaries if s and str(s).strip()]
    return text_summaries, table_summaries

