import uuid
import threading
from src.liars_dice import LiarsDice
from src.batch_runner import GameBatchRunner

class GameManager:
    def __init__(self):
        # Active single games: game_id -> LiarsDice instance
        self.games: dict[str, LiarsDice] = {}
        # Active tournaments: tournament_id -> dict with runner and status
        self.tournaments: dict[str, dict] = {}
        self.lock = threading.Lock()

    # Single game methods

    def create_game(self, players: list[str], ai_models: dict[str, str]) -> str:
        game = LiarsDice()
        for name in players:
            game.add_player(name)
        for name, model in ai_models.items():
            game.add_ai_player(name, model)
        game.roll_all_dice()
        with self.lock:
            game_id = str(uuid.uuid4())
            self.games[game_id] = game
        return game_id

    def process_move(self, game_id: str, action: str, quantity: int = None, value: int = None) -> dict:
        game = self.games.get(game_id)
        if not game:
            raise ValueError("Game not found")
        if action == "bid":
            if quantity is None or value is None:
                raise ValueError("Bid requests require quantity and value")
            game.last_bid = (quantity, value)
            move = {
                "round": game.round_number,
                "player": game.players[game.current_player_idx].name,
                "action": "bid",
                "quantity": quantity,
                "value": value
            }
            game.move_history.append(move)
            game.next_player()
        elif action == "liar":
            game.handle_liar_call(auto_continue=True)
            if not game.check_game_over():
                game.round_number += 1
                game.roll_all_dice()
        else:
            raise ValueError("Invalid action")
        return self._serialize_game(game)

    def get_game_state(self, game_id: str) -> dict | None:
        game = self.games.get(game_id)
        if not game:
            return None
        return self._serialize_game(game)

    def list_games(self) -> list[str]:
        return list(self.games.keys())

    def get_all_stats(self) -> dict:
        return {gid: game.get_metrics() for gid, game in self.games.items()}

    def _serialize_game(self, game: LiarsDice) -> dict:
        return {
            "players": [game._serialize_player(p) for p in game.players],
            "current_player": game.players[game.current_player_idx].name,
            "last_bid": game.last_bid,
            "round_number": game.round_number,
            "move_history": game.move_history,
            "game_over": game.game_over,
            "winner": game.winner.name if game.winner else None,
            "metrics": game.get_metrics()
        }

    # AI models discovery

    def get_available_models(self, use_local: bool = False) -> list[dict]:
        return LiarsDice().get_available_models(use_local, False)

    # Tournament methods

    def create_tournament(self, model_list: list[dict], total_games: int, models_per_game: int) -> str:
        runner = GameBatchRunner()
        # Configure runner with selected models and settings
        runner.selected_models = model_list
        runner.total_games = total_games
        runner.models_per_game = models_per_game
        runner.leaderboard = {m["id"]: {"wins": 0, "games_played": 0, "win_rate": 0.0} for m in model_list}
        runner.model_stats = {}
        for m in model_list:
            mid = m["id"]
            runner.model_stats[mid] = {
                "total_rounds_played": 0,
                "avg_rounds_per_game": 0,
                "total_bids": 0,
                "total_liar_calls": 0,
                "successful_liar_calls": 0,
                "unsuccessful_liar_calls": 0,
                "early_game_wins": 0,
                "mid_game_wins": 0,
                "long_game_wins": 0
            }
        t_id = str(uuid.uuid4())
        self.tournaments[t_id] = {
            "runner": runner,
            "status": "pending",
            "results": None
        }
        return t_id

    def list_tournaments(self) -> list[str]:
        return list(self.tournaments.keys())

    def get_tournament(self, tournament_id: str) -> dict | None:
        info = self.tournaments.get(tournament_id)
        if not info:
            return None
        runner = info["runner"]
        return {
            "settings": {
                "models": runner.selected_models,
                "total_games": runner.total_games,
                "models_per_game": runner.models_per_game
            },
            "status": info["status"],
            "leaderboard": runner.leaderboard
        }

    def start_tournament(self, tournament_id: str) -> None:
        info = self.tournaments.get(tournament_id)
        if not info:
            raise ValueError("Tournament not found")
        if info["status"] != "pending":
            raise ValueError("Tournament already started")
        runner: GameBatchRunner = info["runner"]
        # Run tournament in background thread
        thread = threading.Thread(target=runner.run_tournament, daemon=True)
        info["status"] = "running"
        thread.start()

    def get_tournament_results(self, tournament_id: str) -> dict | None:
        info = self.tournaments.get(tournament_id)
        if not info or info["status"] != "completed":
            return None
        runner: GameBatchRunner = info["runner"]
        return {
            "leaderboard": runner.leaderboard,
            "model_stats": runner.model_stats,
            "game_results": runner.game_results
        }

    def complete_tournament(self, tournament_id: str) -> None:
        info = self.tournaments.get(tournament_id)
        if info:
            info["status"] = "completed"
