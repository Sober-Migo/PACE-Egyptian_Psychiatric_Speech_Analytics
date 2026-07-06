# Application factory and ASGI server entry point
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router import api_router

def create_app() -> FastAPI:
    """Initializes the FastAPI framework and binds modular routes."""
    app = FastAPI(title="PACE - Psychiatric Analytics Core Engine")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app

app = create_app()

if __name__ == "__main__":
    print("\n🚀 PACE Backend initialized. Serving on http://0.0.0.0:8000\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)