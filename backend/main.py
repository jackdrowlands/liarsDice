from fastapi import FastAPI
from .api import router

app = FastAPI(
    title="Liar's Dice API",
    description="API for controlling Liar’s Dice games and tournaments",
    version="0.1.0",
)

app.include_router(router, prefix="/api")
