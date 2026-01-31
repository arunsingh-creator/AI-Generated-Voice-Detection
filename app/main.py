"""
FastAPI application entry point
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import detection_router
from app.config import settings
import logging

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AI Voice Detection API",
    description="REST API for detecting AI-generated vs human voices in multiple languages",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error occurred"
        }
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Voice Detection API",
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "endpoints": {
            "voice_detection": "/api/voice-detection",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        "supported_formats": ["mp3"]
    }


# Register routers
app.include_router(detection_router, tags=["Voice Detection"])


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AI Voice Detection API...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Sample Rate: {settings.SAMPLE_RATE} Hz")
    logger.info(f"Model Path: {settings.MODEL_PATH}")
    
    # Pre-load classifier
    from app.services import get_classifier
    classifier = get_classifier()
    if classifier.use_heuristic:
        logger.warning("⚠ Using heuristic-based classification (no trained model found)")
    else:
        logger.info("✓ Using trained ML model for classification")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Voice Detection API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development"
    )
