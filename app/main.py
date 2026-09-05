from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Path, Request

from app.config import config
from app.schemas import RecommendationItem, RecommendationResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.model = joblib.load(config.MODEL_PATH)

    yield

    app.state.model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    request: Request, user_id: int = Path(..., title="ID пользователя", ge=1)
) -> RecommendationResponse:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    preds = model.recommend_top_n(user_id, n=10)
    recommendations = [
        RecommendationItem(movie_id=row.movieId, title=row.title, score=row.final_score)
        for row in preds.itertuples()
    ]

    return RecommendationResponse(user_id=user_id, recommendations=recommendations)
