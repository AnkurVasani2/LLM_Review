from fastapi import FastAPI
from endpoints import health_router, paper_review_router

app = FastAPI(
    title="Research Paper Review API",
    description="API for analyzing and reviewing research papers using LLM",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(paper_review_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)