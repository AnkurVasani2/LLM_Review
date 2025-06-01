from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy"}