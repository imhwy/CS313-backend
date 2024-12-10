"""
Run Backend Server for CS313
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers.classification import classification_router

# Initialize FastAPI application
app = FastAPI(
    title="CS313 Backend",
    description="This is the backend API for CS313, handling comment classification.",
    version="1.0"
)

# Add middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (update this for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)

# Include API routers
app.include_router(classification_router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
