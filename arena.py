"""Deterministic, paired evaluation of two 5x5 models."""

import argparse
from numbers import Integral
import random

import numpy as np

from game import (
    BLACK, WHITE, EMPTY, FORBIDDEN, N, CELL_COUNT, LINE_INDICES, PERMS,
    apply_move, initial_state, legal_moves, ordered_moves, other_player,
    move_rejection_reason, validate_state, winner,
)
from inference import encode_state, load_net, ncnn_infer


# Keep the names used by existing callers.
legal = legal_moves
_winner = winner


class Player:
    """Model policy with optional immediate-win / immediate-block assistance."""

    def __init__(self, net, enc_type, name, tactical=False, hint=False):
        if enc_type not in ("bit", "two"):
            raise ValueError("enc_type must be 'bit' or 'two'")
        self.net = net
        self.enc_type = enc_type
        self.name = name
        self.tactical = tactical
        self.hint = hint

    def _encode(self, s, me):
        return encode_state(s, me, encoding=self.enc_type)

    def _cnn_logits(self, s, me):
        policy, _ = ncnn_infer(self.net, self._encode(s, me))
        logits = np.asarray(policy, dtype=np.float64).copy()
        if self.hint:
            from az import pattern_bonus
            logits += np.asarray(pattern_bonus(s, me), dtype=np.float64)
        return logits

    def move(self, s, me):
        validate_state(s)
        other_player(me)  # Validate the player even on terminal boards.
        moves = legal(s)
        if winner(s) != EMPTY or not moves:
            return None
        if self.tactical:
            wins, blocks, _ = ordered_moves(s, me)
            if wins:
                return wins[0]
            if blocks:
                return blocks[0]
        logits = self._cnn_logits(s, me)
        return max(moves, key=lambda i: logits[i])

    def _immediate_win(self, s, me):
        wins, _, _ = ordered_moves(s, me)
        return wins[0] if wins else None

    def _opp_win_cell(self, s, me):
        _, blocks, _ = ordered_moves(s, me)
        return blocks[0] if blocks else None


class PlayerAB(Player):
    """Negamax alpha-beta; CNN priors order moves and break equal-value ties.

    Scores belong to the player whose turn is being searched. Heuristics are
    strictly inside (-1, 1), reserving +/-1 for proven wins and losses.
    """

    def __init__(self, net, name, depth=4, enc_type="bit", hint=False):
        if isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 1:
            raise ValueError("depth must be a positive integer")
        super().__init__(net, enc_type, name, tactical=True, hint=hint)
        self.depth = int(depth)
        self.cache = {}

    def _score(self, s, me):
        weights = {2: 1, 3: 3, 4: 10}
        total = 0
        for line in LINE_INDICES:
            own = opponent = 0
            blocked = False
            for i in line:
                cell = (s >> (i * 2)) & 3
                if cell == FORBIDDEN:
                    blocked = True
                    break
                own += cell == me
                opponent += cell != EMPTY and cell != me
            if blocked:
                continue
            if not opponent:
                total += weights.get(own, 0)
            if not own:
                total -= weights.get(opponent, 0)
        # Each line contributes at most 10; no heuristic can equal a result.
        return total / (len(LINE_INDICES) * 10 + 1)

    def _ordered(self, s, me):
        return ordered_moves(s, me)

    def _ab(self, s, me, depth, alpha=-float("inf"), beta=float("inf")):
        key = (s, me, depth)
        alpha_original, beta_original = alpha, beta
        hit = self.cache.get(key)
        if hit is not None:
            value, bound = hit
            if bound == "exact":
                return value
            if bound == "lower":
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value

        result = winner(s)
        if result != EMPTY:
            value = 1.0 if result == me else -1.0
        else:
            wins, blocks, rest = self._ordered(s, me)
            if wins:
                value = 1.0
            elif len(blocks) > 1:
                # With no winning move, one stone cannot stop two kill cells.
                value = -1.0
            elif not blocks and not rest:
                value = 0.0
            elif depth <= 0:
                value = self._score(s, me)
            else:
                moves = blocks if blocks else rest
                opponent = other_player(me)
                best = -1.0
                for move in moves:
                    child = apply_move(s, move, me)
                    value = -self._ab(child, opponent, depth - 1, -beta, -alpha)
                    best = max(best, value)
                    alpha = max(alpha, best)
                    if alpha >= beta or best == 1.0:
                        break
                # A cutoff is a bound, not an exact transposition value.
                if best == 1.0:
                    bound = "exact"  # Global maximum, regardless of window.
                elif best <= alpha_original:
                    bound = "upper"
                elif best >= beta_original:
                    bound = "lower"
                else:
                    bound = "exact"
                self.cache[key] = (best, bound)
                return best

        self.cache[key] = (value, "exact")
        return value

    def move(self, s, me):
        validate_state(s)
        opponent = other_player(me)
        moves = legal(s)
        if winner(s) != EMPTY or not moves:
            return None
        wins, blocks, rest = self._ordered(s, me)
        if wins:
            return wins[0]
        if len(blocks) == 1:
            return blocks[0]
        priors = self._cnn_logits(s, me)
        candidates = blocks if blocks else rest
        ordered = sorted(candidates, key=lambda i: (-priors[i], i))
        self.cache.clear()
        best_value = -float("inf")
        best_move = ordered[0]
        for move in ordered:
            value = -self._ab(
                apply_move(s, move, me), opponent, self.depth - 1,
                -float("inf"), -best_value,
            )
            # Descending prior order already resolves equal search values.
            if value > best_value:
                best_value, best_move = value, move
            if best_value == 1.0:
                break
        return best_move


def _checked_move(s, move, me, source):
    index = int(move) if isinstance(move, Integral) and not isinstance(move, bool) else move
    reason = move_rejection_reason(s, index)
    if reason is not None:
        raise ValueError(f"{source} returned illegal move {move!r} for player {me}: {reason}")
    return apply_move(s, index, me)


def _opening_position(opening):
    s, me = initial_state(), WHITE
    transcript = []
    for move in opening:
        if winner(s) != EMPTY:
            raise ValueError("opening continues after the game has ended")
        s = _checked_move(s, move, me, "opening")
        transcript.append((me, int(move)))
        me = other_player(me)
    return s, me, transcript


def play_game(black, white, verbose=False, opening=None):
    """Play from the mandatory black center and optional subsequent moves."""
    s, me, transcript = _opening_position(() if opening is None else opening)
    while True:
        result = winner(s)
        if result != EMPTY or not legal(s):
            return result, transcript
        player = black if me == BLACK else white
        move = player.move(s, me)
        s = _checked_move(s, move, me, player.name)
        transcript.append((me, int(move)))
        if verbose:
            print(f"{player.name} player={me}: ({move // N}, {move % N})")
        me = other_player(me)


def _canonical_state(s):
    return min(
        sum(((s >> (old * 2)) & 3) << (new * 2) for old, new in enumerate(perm))
        for perm in PERMS
    )


def generate_openings(count, seed=0, plies=4):
    """Generate reproducible nonterminal positions, unique under board symmetry.

    ``plies`` counts placements after the mandatory black center. Each returned
    move prefix is used twice with the models exchanging colors.
    """
    if count < 1:
        raise ValueError("at least one opening pair is required")
    if not 0 <= plies <= CELL_COUNT - 2:
        raise ValueError(f"opening plies must be between 0 and {CELL_COUNT - 2}")
    if plies == 0:
        if count > 1:
            raise ValueError("zero opening plies permits only one independent opening")
        return [()]
    rng = random.Random(seed)
    openings, seen = [], set()
    for _ in range(max(1000, count * 200)):
        s, me = initial_state(), WHITE
        prefix = []
        for _ in range(plies):
            move = rng.choice(legal(s))
            s = apply_move(s, move, me)
            prefix.append(move)
            me = other_player(me)
            if winner(s) != EMPTY:
                break
        if len(prefix) != plies or winner(s) != EMPTY:
            continue
        key = _canonical_state(s)
        if key in seen:
            continue
        seen.add(key)
        openings.append(tuple(prefix))
        if len(openings) == count:
            return openings
    raise ValueError("not enough independent openings; increase --opening-plies or reduce games")


def evaluate(mine, repo, openings, verbose=False):
    stats = {"W": 0, "L": 0, "D": 0, "Wb": 0, "Ww": 0}
    positions, transcripts = set(), set()
    pairs = 0
    for opening in openings:
        opening = tuple(opening)
        positions.add(_canonical_state(_opening_position(opening)[0]))
        pairs += 1
        for black, white, my_color in ((mine, repo, BLACK), (repo, mine, WHITE)):
            result, transcript = play_game(black, white, verbose, opening)
            transcripts.add(tuple(transcript))
            if result == EMPTY:
                stats["D"] += 1
            elif result == my_color:
                stats["W"] += 1
                stats["Wb" if my_color == BLACK else "Ww"] += 1
            else:
                stats["L"] += 1
    stats.update(games=pairs * 2, pairs=pairs, unique_openings=len(positions),
                 unique_transcripts=len(transcripts))
    return stats


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("my_param")
    parser.add_argument("my_bin")
    parser.add_argument("repo_param")
    parser.add_argument("repo_bin")
    parser.add_argument("games", nargs="?", type=int, default=100,
                        help="even number of games; each opening is played with colors swapped")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--raw", action="store_true", help="policy only, for both models")
    mode.add_argument("--ab", action="store_true", help="alpha-beta assistance for both models")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hint", action="store_true", help="pattern bonuses for both models")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.games < 2 or args.games % 2:
        parser.error("games must be a positive even number (at least 2) for paired evaluation")
    if args.depth < 1:
        parser.error("--depth must be positive")
    if args.raw and args.hint:
        parser.error("--raw cannot be combined with --hint")
    try:
        openings = generate_openings(args.games // 2, args.seed, args.opening_plies)
    except ValueError as error:
        parser.error(str(error))
    mine_net = load_net(args.my_param, args.my_bin)
    repo_net = load_net(args.repo_param, args.repo_bin)
    if args.ab:
        mine = PlayerAB(mine_net, "mine(5x5)+ab", args.depth, "bit", args.hint)
        repo = PlayerAB(repo_net, "repo(5x5)+ab", args.depth, "two", args.hint)
    else:
        mine = Player(mine_net, "bit", "mine(5x5)", tactical=not args.raw, hint=args.hint)
        repo = Player(repo_net, "two", "repo(5x5)", tactical=not args.raw, hint=args.hint)
    stats = evaluate(mine, repo, openings, args.verbose)
    print(f"mine={mine.name} vs repo={repo.name}: {stats['games']} games, seed={args.seed}")
    print(f"  {stats['pairs']} color-swapped pairs; {stats['unique_openings']} independent "
          f"openings (up to symmetry); {stats['unique_transcripts']} unique transcripts")
    print(f"  我方胜 {stats['W']} ({stats['W'] / stats['games'] * 100:.0f}%) "
          f" 负 {stats['L']}  平 {stats['D']}")
    print(f"  执黑胜 {stats['Wb']}  执白胜 {stats['Ww']}")


if __name__ == "__main__":
    main()
