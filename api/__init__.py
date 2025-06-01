from .endpoints.health import router as health_router
from .endpoints.paper_review import router as paper_review_router

__all__ = ['health_router', 'paper_review_router']