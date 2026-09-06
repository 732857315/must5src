"""Network, bounded MCTS self-play, and opt-in training entry point."""

import hashlib
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from game import (
    N, CELL_COUNT, BLACK, WHITE, EMPTY, FORBIDDEN, CENTER, CENTER_INDEX, Board,
    LINE_CELLS, LINE_INDICES, NEIGHBORS, PERMS, cell_index, apply_move,
    initial_state, legal_moves, other_player, stone_count, validate_state, winner, winning_moves,
)

SIMULATIONS = 200
C_PUCT = 1.5
TEMP_PLIES = 5
DIRICHLET_EPS = 0.25
DIRICHLET_ALPHA = 0.3
DEVICE = "cpu"


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x * self.se(x)
        return F.relu(x + r)


class GomokuNet5x5(nn.Module):
    """5x5 纯卷积双头网络, 输入为 2bit 棋盘图 (1,1,5,5):
    0=空, 1=我方, 2=对方, 3=边界/禁下。历史权重尚未学习新增的边界语义。"""

    def __init__(self, channels=32, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[SEBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        policy = self.policy_head(x).flatten(1)
        value = self.value_conv(x)
        value = F.adaptive_avg_pool2d(value, 1).flatten(1)
        value = self.value_fc(value).squeeze(-1)
        return policy, value


# Encoding and symmetry share the same cell mapping as every other player.
def encode_board(s, me):
    validate_state(s)
    other_player(me)
    values = [(s >> (2 * i)) & 3 for i in range(CELL_COUNT)]
    return torch.tensor(
        [float(FORBIDDEN) if c == FORBIDDEN else 0.0 if c == EMPTY else 1.0 if c == me else 2.0
         for c in values],
        dtype=torch.float32,
    ).reshape(1, 1, N, N)


def sym_augment(x, pi):
    if tuple(x.shape) != (1, 1, N, N) or tuple(pi.shape) != (CELL_COUNT,):
        raise ValueError("Expected board (1,1,5,5) and policy (25,)")
    out = []
    for perm in PERMS:
        nx, npi = torch.empty_like(x), torch.empty_like(pi)
        nx.reshape(-1)[list(perm)] = x.reshape(-1)
        npi[list(perm)] = pi
        out.append((nx, npi))
    return out


class Node:
    """w / visits is always from the player to move at THIS node."""
    __slots__ = ("prior", "visits", "w", "children")

    def __init__(self, prior):
        self.prior = prior
        self.visits = 0
        self.w = 0.0
        self.children = None

    def expand(self, priors):
        self.children = {i: Node(p) for i, p in enumerate(priors) if p > 0}

    def q(self):
        return self.w / self.visits if self.visits else 0.0


def select_child(node):
    return max(node.children, key=lambda i: (
        -node.children[i].q()
        + C_PUCT * node.children[i].prior * math.sqrt(node.visits)
        / (1 + node.children[i].visits)
    ))


def guarded_moves(board, side):
    wins = winning_moves(board, side)
    if wins:
        return "win", wins
    # The first black move is fixed by this project's opening rule.
    return "free", [CENTER_INDEX] if board == 0 else legal_moves(board)


LINES_THROUGH = tuple(tuple(line for line in LINE_INDICES if i in line) for i in range(CELL_COUNT))
NEIGHBOR_CELLS = NEIGHBORS
_winner_of = winner  # Compatibility for existing callers.


def pattern_bonus(board, side):
    """Reward open threes/fours; a line containing the opponent cannot become five."""
    opp = other_player(side)
    candidates = {
        j for i in range(CELL_COUNT) if ((board >> (2 * i)) & 3) == side
        for j in NEIGHBORS[i] if ((board >> (2 * j)) & 3) == EMPTY
    }
    bonus = [0.0] * CELL_COUNT
    for i in candidates:
        for line in LINES_THROUGH[i]:
            cells = [(board >> (2 * j)) & 3 for j in line]
            if opp in cells or FORBIDDEN in cells:
                continue
            count = cells.count(side) + 1
            bonus[i] += 2.0 if count == 4 else 0.5 if count == 3 else 0.0
    return bonus


def _network_priors(net, board, side, moves):
    x = encode_board(board, side)
    if isinstance(net, nn.Module):
        parameter = next(net.parameters(), None)
        if parameter is not None:
            x = x.to(parameter.device)
    with torch.no_grad():
        logits, value = net(x)
    if logits.numel() != CELL_COUNT or value.numel() != 1:
        raise ValueError("Network must return 25 policy logits and one value")
    if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
        raise ValueError("Network returned nonfinite output")
    value = float(value.item())
    if abs(value) > 1.000001:
        raise ValueError("Network value must be within [-1, 1]")
    bonus = pattern_bonus(board, side)
    logits = logits.detach().cpu().reshape(-1).tolist()
    scores = [logits[i] + bonus[i] for i in moves]
    peak = max(scores)
    weights = [max(math.exp(score - peak), 1e-12) for score in scores]
    total = sum(weights)
    priors = [0.0] * CELL_COUNT
    for i, weight in zip(moves, weights):
        priors[i] = weight / total
    return priors, value


def _root_noise(root, rng):
    children = list(root.children.values())
    noise = [rng.gammavariate(DIRICHLET_ALPHA, 1.0) for _ in children]
    total = sum(noise)
    for child, value in zip(children, noise):
        fraction = value / total if total else 1.0 / len(children)
        child.prior = (1.0 - DIRICHLET_EPS) * child.prior + DIRICHLET_EPS * fraction


def mcts(net, s, me, sims, noise=False, rng=None):
    """Return a normalized policy; a terminal position has an all-zero policy."""
    validate_state(s)
    other_player(me)
    if not isinstance(sims, int) or isinstance(sims, bool) or sims < 1:
        raise ValueError("sims must be a positive integer")
    if getattr(net, "training", False):
        raise ValueError("Call net.eval() before MCTS")
    if winner(s) != EMPTY or not legal_moves(s):
        return [0.0] * CELL_COUNT
    rng = random if rng is None else rng
    root = Node(1.0)
    for _ in range(sims):
        node, board, side = root, s, me
        path = [root]
        while node.children is not None:
            idx = select_child(node)
            board |= side << (2 * idx)
            side = other_player(side)
            node = node.children[idx]
            path.append(node)
        won = winner(board)
        if won != EMPTY:
            value = 1.0 if won == side else -1.0
        elif not legal_moves(board):
            value = 0.0
        else:
            kind, moves = guarded_moves(board, side)
            if kind == "win":
                priors = [1.0 / len(moves) if i in moves else 0.0 for i in range(CELL_COUNT)]
                value = 1.0
            else:
                priors, value = _network_priors(net, board, side, moves)
            node.expand(priors)
            if node is root and noise:
                _root_noise(root, rng)
        for visited in reversed(path):
            visited.visits += 1
            visited.w += value
            value = -value
    # Root expansion itself is not a child visit. For one simulation use priors.
    visits = sum(child.visits for child in root.children.values())
    pi = [0.0] * CELL_COUNT
    for i, child in root.children.items():
        pi[i] = child.visits / visits if visits else child.prior
    return pi


def choose_move(pi, temp, rng=None):
    if len(pi) != CELL_COUNT or any(not math.isfinite(p) or p < 0 for p in pi):
        raise ValueError("Policy must contain 25 finite nonnegative weights")
    if not math.isfinite(temp) or temp < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    candidates = [i for i, p in enumerate(pi) if p > 0]
    if not candidates:
        raise ValueError("Cannot choose a move from an empty policy")
    rng = random if rng is None else rng
    if temp <= 0.05:
        peak = max(pi)
        return rng.choice([i for i in candidates if pi[i] == peak])
    # Scaling by the maximum prevents overflow for small positive temperatures.
    peak = max(pi)
    weights = [(pi[i] / peak) ** (1.0 / temp) for i in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def self_play_game(net, rng, simulations=None):
    s, me = initial_state(), WHITE
    positions = []
    while winner(s) == EMPTY and legal_moves(s):
        pi = mcts(net, s, me, SIMULATIONS if simulations is None else simulations, noise=True, rng=rng)
        temp = 1.0 if stone_count(s) < TEMP_PLIES else 0.0
        move = choose_move(pi, temp, rng)
        positions.append((s, me, pi))
        s = apply_move(s, move, me)
        me = other_player(me)
    won = winner(s)
    samples = []
    for board, side, pi in positions:
        z = 0.0 if won == EMPTY else 1.0 if won == side else -1.0
        for x, target in sym_augment(encode_board(board, side), torch.tensor(pi, dtype=torch.float32)):
            samples.append((x, target, z))
    return samples


def replay_partition(x):
    """Stable split by canonical position: rotated/duplicate boards never leak across sets."""
    cells = x.detach().cpu().reshape(-1).tolist()
    if len(cells) != CELL_COUNT or any(c not in (EMPTY, BLACK, WHITE, FORBIDDEN) for c in cells):
        raise ValueError("Replay board must have 25 cells encoded as 0, 1, 2, or 3 (forbidden)")
    key = min(bytes(int(cells[i]) for i in perm) for perm in PERMS)
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % 10


def split_replay(buffer):
    train_indices, validation_indices = [], []
    for i, (x, _, _) in enumerate(buffer):
        (validation_indices if replay_partition(x) == 0 else train_indices).append(i)
    return train_indices, validation_indices


def train(net, opt, buffer, rng=None):
    if len(buffer) < 512:
        return 0.0, 0.0, 0.0
    rng = random if rng is None else rng
    tr, va = split_replay(buffer)
    if not tr or not va:
        raise ValueError("Replay needs distinct training and validation position groups")
    device = next(net.parameters()).device
    xs = torch.cat([x for x, _, _ in buffer]).to(device)
    pis = torch.stack([pi for _, pi, _ in buffer]).to(device)
    zs = torch.tensor([z for _, _, z in buffer], dtype=torch.float32, device=device)
    net.train()
    total_loss, steps = 0.0, 0
    for _ in range(2):
        rng.shuffle(tr)
        for i in range(0, len(tr), 256):
            idx = tr[i:i + 256]
            logits, value = net(xs[idx])
            loss = -(pis[idx] * F.log_softmax(logits, dim=1)).sum(dim=1).mean() + F.mse_loss(value, zs[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            steps += 1
    net.eval()
    with torch.no_grad():
        logits, value = net(xs[va])
        top1 = (logits.argmax(1) == pis[va].argmax(1)).float().mean().item()
        # Treat values near zero as draws, rather than marking every draw incorrect.
        predicted = torch.where(value.abs() <= 0.25, 0.0, value.sign())
        value_accuracy = (predicted == zs[va]).float().mean().item()
    return total_loss / max(1, steps), top1, value_accuracy


def main():
    seed = int(os.environ.get("SEED", "42"))
    iters = int(os.environ.get("ITERS", "30"))
    games_per_iter = int(os.environ.get("GAMES", "12"))
    if iters < 1 or games_per_iter < 1:
        raise ValueError("ITERS and GAMES must be positive")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = GomokuNet5x5().to(DEVICE).eval()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = []
    t0 = time.time()
    for it in range(1, iters + 1):
        for _ in range(games_per_iter):
            buffer.extend(self_play_game(net, rng))
        buffer = buffer[-40000:]
        loss, top1, value_accuracy = train(net, opt, buffer, rng)
        elapsed = (time.time() - t0) / 60
        print(f"[{elapsed:5.1f}min] iter {it}/{iters} | games={it * games_per_iter} "
              f"samples={len(buffer)} loss={loss:.3f} top1={top1:.3f} value_acc={value_accuracy:.3f}", flush=True)
        if it % 10 == 0:
            torch.save(net.state_dict(), f"gomoku5x5_it{it}.pt")
    torch.save(net.state_dict(), "gomoku5x5_final.pt")
    print(f"done, params={sum(p.numel() for p in net.parameters()):,}, total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
