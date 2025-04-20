from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .game_manager import GameManager

router = APIRouter()
manager = GameManager()

class StartGameRequest(BaseModel):
    players: list[str]
    ai_models: dict[str, str] = {}

class MoveRequest(BaseModel):
    game_id: str
    action: str  # "bid" or "liar"
    quantity: int | None = None
    value: int | None = None

@router.post("/start-game")
def start_game(req: StartGameRequest):
    game_id = manager.create_game(req.players, req.ai_models)
    return {"game_id": game_id}

@router.post("/make-move")
def make_move(req: MoveRequest):
    try:
        state = manager.process_move(
            req.game_id, req.action, quantity=req.quantity, value=req.value
        )
        return {"state": state}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status/{game_id}")
def get_status(game_id: str):
    state = manager.get_game_state(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")
    return {"state": state}

@router.get("/stats")
def get_stats():
    return {"stats": manager.get_all_stats()}

@router.get("/games")
def list_games():
    return {"games": manager.list_games()}

@router.get("/ai-models")
def get_ai_models(use_local: bool = False):
    try:
        models = manager.get_available_models(use_local)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ModelInfo(BaseModel):
    id: str
    provider: str
    name: str
    api_key: str | None = None
    api_url: str | None = None
    project_id: str | None = None
    region: str | None = None

class TournamentCreateRequest(BaseModel):
    models: list[ModelInfo]
    total_games: int
    models_per_game: int

@router.post("/tournaments")
def create_tournament(req: TournamentCreateRequest):
    try:
        model_list = [m.dict() for m in req.models]
        tid = manager.create_tournament(model_list, req.total_games, req.models_per_game)
        return {"tournament_id": tid}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/tournaments")
def list_tournaments():
    return {"tournaments": manager.list_tournaments()}

@router.get("/tournaments/{tournament_id}")
def get_tournament(tournament_id: str):
    info = manager.get_tournament(tournament_id)
    if not info:
        raise HTTPException(status_code=404, detail="Tournament not found")
    return info

@router.post("/tournaments/{tournament_id}/start")
def start_tournament(tournament_id: str):
    try:
        manager.start_tournament(tournament_id)
        return {"status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/tournaments/{tournament_id}/results")
def get_tournament_results(tournament_id: str):
    results = manager.get_tournament_results(tournament_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Results not available")
    return results
