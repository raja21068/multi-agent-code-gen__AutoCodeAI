"""
main.py — FastAPI application entry point.

Run:
    uvicorn main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from api.routes import router, get_orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_orchestrator()          # warm up on startup
    yield
    from api.routes import _orchestrator
    if _orchestrator:
        _orchestrator.shutdown()


app = FastAPI(
    title="AI Coder",
    description="Multi-agent autonomous coding system with sandboxed execution.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
