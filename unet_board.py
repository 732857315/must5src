"""Batch both U-Nets over stone-centered 5x5 views of any rectangular board.

Raw window fusion, immediate full-board tactics, and optional external search
are separate decisions. This module never applies a search or invents a forced
win beyond a verified one-move win.
"""

import heapq
from numbers import Integral

import numpy as np
import torch
from torch import nn

from board_rules import normalize_board, board_winner, winning_cells
from board_threats import analyze_threats
from game import N, BLACK, WHITE, EMPTY, FORBIDDEN, legal_moves
from unet_codec import encode_rgb, policy_rgb
from unet_models import masked_probabilities
from windows import BoardWindow


WINDOW_RADIUS = N // 2
WINDOW_BATCH_SIZE = 128
FUSION = {
    "method": "mean_relative_to_local_uniform",
    "description": "每窗概率乘本窗合法空格数，按真实格覆盖次数平均，再在全盘合法空格归一化",
    "probability_interpretation": "重叠局部模型的归一化偏好，不是校准的全盘落子频率",
}


def _bounds(center, shape):
    row, col = center
    height, width = shape
    return (max(0, row - WINDOW_RADIUS), min(height, row + WINDOW_RADIUS + 1),
            max(0, col - WINDOW_RADIUS), min(width, col + WINDOW_RADIUS + 1))


def _select_centers(board):
    """Keep every stone center; greedily add centers until every empty cell is seen.

    The lazy heap uses linear storage. Gains only decrease, and ties use row/
    column order. This is deterministic greedy coverage, not a claim that the
    number of additional windows is globally minimal.
    """
    stone_centers = [tuple(map(int, coordinate)) for coordinate in np.argwhere(
        (board == BLACK) | (board == WHITE))]
    primary = stone_centers or [(board.shape[0] // 2, board.shape[1] // 2)]
    uncovered = board == EMPTY
    for center in primary:
        top, bottom, left, right = _bounds(center, board.shape)
        uncovered[top:bottom, left:right] = False
    remaining = int(uncovered.sum())
    extras = []
    if remaining:
        # Summed-area gains initialize one heap entry per useful empty center.
        integral = np.pad(uncovered.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        height, width = board.shape
        rows, cols = np.arange(height), np.arange(width)
        top, bottom = np.maximum(0, rows - WINDOW_RADIUS), np.minimum(height, rows + WINDOW_RADIUS + 1)
        left, right = np.maximum(0, cols - WINDOW_RADIUS), np.minimum(width, cols + WINDOW_RADIUS + 1)
        gains = (integral[bottom[:, None], right[None, :]] - integral[top[:, None], right[None, :]]
                 - integral[bottom[:, None], left[None, :]] + integral[top[:, None], left[None, :]])
        heap = [(-int(gains[row, col]), int(row), int(col))
                for row, col in np.argwhere((board == EMPTY) & (gains > 0))]
        heapq.heapify(heap)
        while remaining:
            if not heap:
                raise AssertionError("An empty cell has no covering window center")
            negative_bound, row, col = heapq.heappop(heap)
            top, bottom, left, right = _bounds((row, col), board.shape)
            gain = int(uncovered[top:bottom, left:right].sum())
            if gain == 0:
                continue
            if gain != -negative_bound:
                heapq.heappush(heap, (-gain, row, col))
                continue
            extras.append((row, col))
            uncovered[top:bottom, left:right] = False
            remaining -= gain
    return stone_centers, list(primary), extras


def _extract_normalized(board, center):
    """Extract a window in constant work, without revalidating the whole board."""
    row, col = center
    top, bottom, left, right = _bounds(center, board.shape)
    local = np.full((N, N), FORBIDDEN, dtype=np.uint8)
    local_top, local_left = top - row + WINDOW_RADIUS, left - col + WINDOW_RADIUS
    local[local_top:local_top + bottom - top, local_left:local_left + right - left] = board[top:bottom, left:right]
    state = sum(int(cell) << (2 * index) for index, cell in enumerate(local.reshape(-1)))
    height, width = board.shape
    coordinates = tuple(
        (row + index // N - WINDOW_RADIUS, col + index % N - WINDOW_RADIUS)
        if (0 <= row + index // N - WINDOW_RADIUS < height
            and 0 <= col + index % N - WINDOW_RADIUS < width) else None
        for index in range(N * N)
    )
    return BoardWindow(state, center, coordinates)


def _forward_probabilities(model, images, side, states):
    if isinstance(model, nn.Module):
        model.eval()
    rgb = torch.from_numpy(np.stack(images))
    if isinstance(model, nn.Module):
        parameter = next(model.parameters(), None)
        if parameter is not None:
            rgb = rgb.to(device=parameter.device, dtype=parameter.dtype)
    sides = torch.full((len(states),), side, dtype=torch.long, device=rgb.device)
    logits = model(rgb, sides)
    return masked_probabilities(logits, states)[:, 0].reshape(-1, N * N).cpu().numpy()


def _normalize_fusion(total, coverage, legal):
    if np.any(coverage[legal] == 0):
        raise AssertionError("A real empty cell was not covered by an active window")
    fused = np.zeros(total.shape, dtype=np.float64)
    fused[legal] = total[legal] / coverage[legal]
    mass = float(fused.sum())
    if not np.isfinite(fused).all() or np.any(fused < 0) or not np.isfinite(mass) or mass <= 0:
        raise ValueError("Window fusion did not produce finite positive legal probability mass")
    fused /= mass
    return fused


def _restricted_policy(raw, moves):
    result = np.zeros_like(raw)
    rows, cols = zip(*moves)
    result[rows, cols] = raw[rows, cols]
    mass = float(result.sum())
    if mass > 0:
        result /= mass
    else:
        result[rows, cols] = 1.0 / len(moves)
    return result


def analyze_board(grid, my_side, opponent_model, play_model, *, tactical=True):
    """Analyze any HxW grid using explicit global color and bounded window batches.

    Policies are float64 HxW arrays; coverage counts only real empty cells.
    A terminal board returns the same fields with zero arrays and no model call.
    tactical=False returns the raw network recommendation without any override.
    """
    if (isinstance(my_side, bool) or not isinstance(my_side, Integral)
            or my_side not in (BLACK, WHITE)):
        raise ValueError("my_side必须明确为1黑棋或2白棋，不能从局部棋子数推断")
    if not isinstance(tactical, bool):
        raise ValueError("tactical must be a boolean")
    my_side = int(my_side)
    board = normalize_board(grid)
    legal = board == EMPTY
    won = board_winner(board)
    coverage = np.zeros(board.shape, dtype=np.int32)
    zeros = np.zeros(board.shape, dtype=np.float64)
    result = {
        "opponent_policy": zeros.copy(), "play_policy": zeros.copy(),
        "raw_play_policy": zeros.copy(), "move": None, "raw_move": None,
        "reason": "", "window_count": 0, "coverage": coverage,
        "terminal": bool(won != EMPTY or not legal.any()), "winner": int(won),
        "my_side": my_side, "tactical_enabled": tactical, "tactical_applied": False,
        "tactical_kind": None, "stone_centers": [], "extra_centers": [],
        "window_centers": [], "skipped_window_centers": [], "model_batches": 0,
        "window_batch_size": WINDOW_BATCH_SIZE, "fusion": dict(FUSION),
        "threats": {"self": analyze_threats(board, my_side),
                    "opponent": analyze_threats(board, 3 - my_side)},
    }
    if result["terminal"]:
        result["reason"] = (f"棋局已结束：{'黑方' if won == BLACK else '白方'}已连成五子或以上"
                            if won else "棋局已结束：没有可落子的空位")
        return result

    stone_centers, primary, extras = _select_centers(board)
    centers = primary + extras
    result["stone_centers"], result["extra_centers"] = stone_centers, extras
    opponent_sum, play_sum = zeros.copy(), zeros.copy()
    with torch.inference_mode():
        for start in range(0, len(centers), WINDOW_BATCH_SIZE):
            windows = []
            moves_by_window = []
            for center in centers[start:start + WINDOW_BATCH_SIZE]:
                window = _extract_normalized(board, center)
                moves = legal_moves(window.state)
                if not moves:
                    result["skipped_window_centers"].append(center)
                    continue
                windows.append(window)
                moves_by_window.append(moves)
            if not windows:
                continue
            states = [window.state for window in windows]
            opponent = _forward_probabilities(
                opponent_model, [encode_rgb(state) for state in states], 3 - my_side, states)
            play = _forward_probabilities(
                play_model, [policy_rgb(state, prediction, "red", levels=None)
                             for state, prediction in zip(states, opponent)], my_side, states)
            result["model_batches"] += 1
            for window, moves, red, green in zip(windows, moves_by_window, opponent, play):
                coordinates = [window.coordinates[index] for index in moves]
                if any(coordinate is None for coordinate in coordinates):
                    raise AssertionError("Forbidden padding escaped the local action mask")
                rows, cols = zip(*coordinates)
                # A uniform local predictor contributes preference 1 regardless
                # of corner padding, forbidden cells, or occupied-cell count.
                local_count = len(moves)
                opponent_sum[rows, cols] += red[moves].astype(np.float64) * local_count
                play_sum[rows, cols] += green[moves].astype(np.float64) * local_count
                coverage[rows, cols] += 1
                result["window_centers"].append(window.center)
    result["window_count"] = len(result["window_centers"])
    opponent_policy = _normalize_fusion(opponent_sum, coverage, legal)
    raw_play = _normalize_fusion(play_sum, coverage, legal)
    play_policy = raw_play.copy()
    raw_move = tuple(map(int, np.unravel_index(raw_play.argmax(), board.shape)))
    result["reason"] = "采用重叠窗口融合后的网络最高偏好落点"
    if tactical:
        wins = winning_cells(board, my_side)
        if wins:
            play_policy = _restricted_policy(raw_play, wins)
            result.update(tactical_applied=True, tactical_kind="immediate_win",
                          reason="全盘一步成五：优先选择已验证的立即获胜点")
        else:
            blocks = winning_cells(board, 3 - my_side)
            if len(blocks) == 1:
                play_policy = _restricted_policy(raw_play, blocks)
                result.update(tactical_applied=True, tactical_kind="unique_defense",
                              reason="全盘唯一防守点：阻止对手下一步连成五子")
            elif len(blocks) > 1:
                result["reason"] = "对手存在多个一步胜点；当前返回网络融合建议，未作搜索证明"
    move = tuple(map(int, np.unravel_index(play_policy.argmax(), board.shape)))
    result.update(opponent_policy=opponent_policy, raw_play_policy=raw_play,
                  play_policy=play_policy, move=move, raw_move=raw_move)
    return result