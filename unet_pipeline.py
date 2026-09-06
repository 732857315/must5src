"""Load the two trained U-Nets and generate legal red/green 5x5 predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from game import BLACK, WHITE, Board, apply_move, other_player, validate_state
from unet_codec import encode_rgb, policy_rgb, normalize_grid
from unet_models import OpponentUNet5x5, PlayUNet5x5, masked_probabilities


CHECKPOINT_FORMAT = "gomoku_rgb_unet_v1"


def load_model(path, role):
    if role not in ("opponent", "play"):
        raise ValueError("role must be opponent or play")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a versioned RGB U-Net checkpoint, not an old SE-ResNet weight file")
    if payload.get("role") != role:
        raise ValueError(f"Expected {role} weights; received {payload.get('role')}")
    model_class = OpponentUNet5x5 if role == "opponent" else PlayUNet5x5
    model = model_class(base_channels=payload["base_channels"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Checkpoint has nonfinite weights")
    model.eval()
    return model, {key: value for key, value in payload.items() if key != "state_dict"}


def predict_opponent(state, my_side, opponent_model):
    """Predict the opponent if it moves on THIS position, before applying any move."""
    validate_state(state)
    opponent_side = other_player(my_side)
    with torch.inference_mode():
        logits = opponent_model(torch.from_numpy(encode_rgb(state))[None], torch.tensor([opponent_side]))
        return masked_probabilities(logits, [state])[0, 0].cpu().numpy()


def analyze_position(state, my_side, opponent_model, play_model):
    """The play model consumes the opponent model's continuous red probability image."""
    opponent = predict_opponent(state, my_side, opponent_model)
    rgb = policy_rgb(state, opponent, "red", levels=None)
    with torch.inference_mode():
        logits = play_model(torch.from_numpy(rgb)[None], torch.tensor([my_side]))
        play = masked_probabilities(logits, [state])[0, 0].cpu().numpy()
    move = int(play.argmax()) if play.sum() > 0 else None
    return {"state": state, "my_side": my_side, "opponent_policy": opponent,
            "play_policy": play, "move": move,
            "opponent_assumption": "opponent_to_move_on_current_position"}


def predict_reply(state, my_side, move, opponent_model):
    """For an actual opponent reply, first apply the specified own move."""
    next_state = apply_move(state, move, my_side)
    return next_state, predict_opponent(next_state, my_side, opponent_model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="directory containing opponent.pt and play.pt")
    parser.add_argument("--board", default="examples/board_5x5.json")
    parser.add_argument("--side", choices=("black", "white"), required=True)
    parser.add_argument("--output", default="exports/unet_prediction")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    state = Board(normalize_grid(json.loads(Path(args.board).read_text(encoding="utf-8-sig"))).reshape(-1).tolist()).pack()
    directory = Path(args.models)
    opponent, opponent_meta = load_model(directory / "opponent.pt", "opponent")
    play, play_meta = load_model(directory / "play.pt", "play")
    result = analyze_position(state, BLACK if args.side == "black" else WHITE, opponent, play)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plain = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}
    plain["models"] = {"opponent": opponent_meta, "play": play_meta}
    output.with_suffix(".json").write_text(json.dumps(plain, ensure_ascii=False, indent=2), encoding="utf-8")
    from unet_visuals import render_html, render_svg, render_png
    trained = bool(opponent_meta.get('trained') and play_meta.get('trained'))
    options = dict(
        status="已训练模型推理" if trained else "未确认训练状态的模型推理",
        subtitle=f"己方：{'黑棋' if args.side == 'black' else '白棋'}；红图假设对手在当前局面落子，绿图为己方落子推荐",
        checkpoint_source=str(directory.resolve()),
        training_stats={"去重基础样本": play_meta.get('unique_base_samples', '未知'),
                        "课程阶段": play_meta.get('stage', '未知')},
    )
    policies = (state, result['opponent_policy'], result['play_policy'])
    output.with_suffix('.html').write_text(render_html(*policies, **options), encoding='utf-8')
    output.with_suffix('.svg').write_text(render_svg(*policies, **options), encoding='utf-8')
    output.with_suffix('.png').write_bytes(render_png(*policies, **options))
    print(json.dumps({"move": result["move"], "html": str(output.with_suffix('.html')),
                      "json": str(output.with_suffix('.json'))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
