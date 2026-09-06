"""Regression tests for rules and inference-only MCTS; no optimizer/training calls."""

import random
import unittest
from pathlib import Path

import torch

import az
from game import (
    BLACK, WHITE, EMPTY, FORBIDDEN, CENTER_INDEX, LINE_INDICES, PERMS, Board,
    apply_move, initial_state, legal_moves, player_at, validate_state, winner,
)


class ZeroNet:
    def __call__(self, x):
        return torch.zeros(1, 25), torch.zeros(1)


def threat_position():
    return sum(BLACK << (2 * i) for i in [6, 10, 12, 17]) | sum(WHITE << (2 * i) for i in [0, 1, 2, 3])


class RulesTests(unittest.TestCase):
    def test_all_twelve_winning_lines(self):
        self.assertEqual(len(LINE_INDICES), 12)
        for line in LINE_INDICES:
            for side in (BLACK, WHITE):
                state = sum(side << (2 * i) for i in line)
                self.assertEqual(winner(state), side)
                self.assertEqual(Board.unpack(state).winner(), side)

    def test_roundtrip_and_reference_winner_on_legal_games(self):
        rng = random.Random(19)
        for _ in range(30):
            s = initial_state()
            while True:
                board = Board.unpack(s)
                expected = next((board.grid[line[0]] for line in LINE_INDICES
                                 if board.grid[line[0]] and len({board.grid[i] for i in line}) == 1), EMPTY)
                self.assertEqual(winner(s), expected)
                self.assertEqual(board.pack(), s)
                if winner(s) or not legal_moves(s):
                    break
                s = apply_move(s, rng.choice(legal_moves(s)), player_at(s))

    def test_invalid_moves_and_states_are_rejected(self):
        for state in (-1, 1 << 50):
            with self.assertRaises(ValueError):
                validate_state(state)
            with self.assertRaises(ValueError):
                Board.unpack(state)
        for idx in (-1, 25, CENTER_INDEX):
            with self.assertRaises(ValueError):
                apply_move(initial_state(), idx, WHITE)
        won = sum(BLACK << (2 * i) for i in range(5))
        with self.assertRaises(ValueError):
            apply_move(won, 9, WHITE)

    def test_constructor_copies_grid(self):
        cells = bytearray(25)
        board = Board(cells)
        cells[0] = BLACK
        self.assertEqual(board.get(0, 0), EMPTY)


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_child_value_is_from_opponents_perspective(self):
        root = az.Node(1.0)
        root.expand([0.5, 0.5])
        root.visits = 20
        for child in root.children.values():
            child.visits = 10
        root.children[0].w = -10
        root.children[1].w = 10
        self.assertEqual(az.select_child(root), 0)

    def test_mcts_finds_unique_block(self):
        pi = az.mcts(ZeroNet(), threat_position(), BLACK, 200)
        self.assertEqual(max(range(25), key=lambda i: pi[i]), 4)
        self.assertGreater(pi[4], 0.8)
        self.assertAlmostEqual(sum(pi), 1.0)

    def test_immediate_win_and_forced_first_move(self):
        s = sum(BLACK << (2 * i) for i in range(4)) | sum(WHITE << (2 * i) for i in (5, 8, 12, 16))
        for simulations in (1, 2, 20):
            pi = az.mcts(ZeroNet(), s, BLACK, simulations)
            self.assertEqual(pi[4], 1.0)
        pi = az.mcts(ZeroNet(), 0, BLACK, 1)
        self.assertEqual(pi[CENTER_INDEX], 1.0)

    def test_terminal_empty_policy_and_bad_budget(self):
        won = sum(BLACK << (2 * i) for i in range(5))
        self.assertEqual(az.mcts(ZeroNet(), won, WHITE, 1), [0.0] * 25)
        with self.assertRaises(ValueError):
            az.choose_move([0.0] * 25, 0)
        for sims in (0, -1, True):
            with self.assertRaises(ValueError):
                az.mcts(ZeroNet(), initial_state(), WHITE, sims)

    def test_root_noise_works_and_is_reproducible(self):
        args = (ZeroNet(), initial_state(), WHITE, 1)
        quiet = az.mcts(*args)
        noisy = az.mcts(*args, noise=True, rng=random.Random(5))
        self.assertNotEqual(quiet, noisy)
        self.assertEqual(noisy, az.mcts(*args, noise=True, rng=random.Random(5)))
        self.assertAlmostEqual(sum(noisy), 1.0)
        self.assertEqual(noisy[CENTER_INDEX], 0.0)

    def test_extreme_logits_stay_finite(self):
        class ExtremeNet:
            def __call__(self, x):
                return torch.arange(25, dtype=torch.float32).reshape(1, 25) * 10000, torch.zeros(1)
        pi = az.mcts(ExtremeNet(), initial_state(), WHITE, 1)
        self.assertAlmostEqual(sum(pi), 1.0)
        self.assertTrue(all(p > 0 for i, p in enumerate(pi) if i != CENTER_INDEX))
        class InvalidNet:
            def __call__(self, x):
                return torch.full((1, 25), float('nan')), torch.zeros(1)
        with self.assertRaises(ValueError):
            az.mcts(InvalidNet(), initial_state(), WHITE, 1)

    def test_self_play_has_only_legal_policy_targets(self):
        # Fake network and four simulations: just exercise game logic, never train.
        samples = az.self_play_game(ZeroNet(), random.Random(42), simulations=4)
        repeated = az.self_play_game(ZeroNet(), random.Random(42), simulations=4)
        self.assertEqual(len(samples), len(repeated))
        first_x, first_pi, _ = samples[0]
        self.assertEqual(first_x.reshape(-1)[CENTER_INDEX].item(), 2.0)
        self.assertEqual(first_pi[CENTER_INDEX].item(), 0.0)
        for (x, pi, z), (rx, rpi, rz) in zip(samples, repeated):
            self.assertAlmostEqual(pi.sum().item(), 1.0, places=6)
            self.assertTrue(torch.all(pi[x.reshape(-1) != 0] == 0))
            self.assertIn(z, (-1.0, 0.0, 1.0))
            self.assertTrue(torch.equal(x, rx) and torch.equal(pi, rpi) and z == rz)

    def test_blocked_lines_get_no_pattern_bonus(self):
        s = sum(BLACK << (2 * i) for i in (0, 1, 2)) | (WHITE << (2 * 4))
        self.assertEqual(az.pattern_bonus(s, BLACK)[3], 0.0)

    def test_forbidden_encoding_is_preserved_and_never_selected(self):
        state = initial_state() | FORBIDDEN | (WHITE << 14)
        for me in (BLACK, WHITE):
            self.assertEqual(az.encode_board(state, me).reshape(-1)[0].item(), FORBIDDEN)
        pi = az.mcts(ZeroNet(), state, BLACK, 40)
        self.assertEqual(pi[0], 0.0)
        self.assertEqual(pi[7], 0.0)
        self.assertEqual(pi[CENTER_INDEX], 0.0)
        self.assertAlmostEqual(sum(pi), 1.0)
        x = az.encode_board(state, BLACK)
        for nx, target in az.sym_augment(x, torch.tensor(pi)):
            self.assertTrue(torch.all(target[nx.reshape(-1) == FORBIDDEN] == 0))
            self.assertEqual(az.replay_partition(nx), az.replay_partition(x))

    def test_forbidden_cells_block_patterns_and_do_not_form_wins(self):
        state = sum(BLACK << (2 * i) for i in (0, 1, 2)) | (FORBIDDEN << 8)
        self.assertEqual(az.pattern_bonus(state, BLACK)[3], 0.0)
        blocked = sum(FORBIDDEN << (2 * i) for i in range(25))
        self.assertEqual(winner(blocked), EMPTY)
        self.assertEqual(az.mcts(ZeroNet(), blocked, BLACK, 10), [0.0] * 25)

    def test_symmetry_targets_and_validation_partition(self):
        x = az.encode_board(threat_position(), BLACK)
        pi = torch.arange(25, dtype=torch.float32)
        samples = [(nx, np, 0.0) for nx, np in az.sym_augment(x, pi)]
        for perm, (nx, np, _) in zip(PERMS, samples):
            for i, j in enumerate(perm):
                self.assertEqual(nx.reshape(-1)[j], x.reshape(-1)[i])
                self.assertEqual(np[j], pi[i])
            self.assertEqual(az.replay_partition(nx), az.replay_partition(x))
        tr, va = az.split_replay(samples * 2)
        self.assertTrue(not tr or not va)
        self.assertEqual(len(tr) + len(va), 16)

    def test_saved_model_architecture_remains_compatible(self):
        checkpoint = Path(__file__).resolve().parents[1] / 'gomoku5x5_final.pt'
        if not checkpoint.exists():
            self.skipTest('Historical checkpoint is not present')
        net = az.GomokuNet5x5().eval()
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        with torch.no_grad():
            policy, value = net(az.encode_board(initial_state(), WHITE))
        self.assertEqual(tuple(policy.shape), (1, 25))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertTrue(torch.isfinite(policy).all() and torch.isfinite(value).all())


if __name__ == '__main__':
    unittest.main()
