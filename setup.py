from setuptools import setup, find_packages

setup(
    name="LLM_review",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "langchain",
        "langchain-groq",
        "langchain-core",
        "langchain-chroma",
        "langchain-huggingface",
        "chromadb",
        "unstructured"
    ]
)