"""
Routes package initialization
"""
from app.routes.detection import router as detection_router

__all__ = ["detection_router"]
