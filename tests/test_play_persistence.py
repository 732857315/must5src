import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from board_rules import board_winner
from play_unet import GameSession, APIError


def analyze(board, side, opponent, play, tactical=True):
    legal = board == 0
    terminal = bool(board_winner(board) or not legal.any())
    p = legal.astype(float)
    if terminal:
        p.fill(0)
    else:
        p /= p.sum()
    return dict(opponent_policy=p, play_policy=p, raw_play_policy=p,
                coverage=np.zeros_like(board) if terminal else legal.astype(int),
                move=None if terminal else tuple(map(int, np.argwhere(legal)[0])))


class GamePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gomoku-persistence-")
        self.directory = Path(self.temporary.name).resolve()
        assert self.directory.is_relative_to(Path(tempfile.gettempdir()).resolve())
        self.file = self.directory / "state.json"
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            session.close()
        self.temporary.cleanup()

    def session(self, **kwargs):
        s = GameSession(None, None, analyzer=analyze, state_file=self.file,
                        persistence_identity=kwargs.pop("identity", {"run_id": "test", "search": {"depth": 5}}), **kwargs)
        self.sessions.append(s)
        return s

    def test_restart_keeps_history_board_revision_and_actor(self):
        first = self.session()
        after = first.move({"row": 1, "col": 1, "revision": first.revision})
        first.close()
        restored = self.session().snapshot()
        for field in ("board", "history", "revision", "human_side", "ai_side", "session"):
            self.assertEqual(restored[field], after[field])
        self.assertEqual(restored["analysis"]["side"], restored["human_side"])

    def test_state_file_is_exclusively_owned(self):
        self.session()
        with self.assertRaises(OSError):
            self.session()

    def test_failed_turn_requires_same_pending_move_even_after_restart(self):
        def fail(*args):
            raise RuntimeError("simulated interruption")
        first = self.session(move_selector=fail)
        original = first.snapshot()
        move = {"row": 1, "col": 1, "revision": first.revision}
        with self.assertLogs(level="ERROR"), self.assertRaises(APIError):
            first.move(move)
        persisted = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(persisted["pending"], move)
        self.assertEqual(persisted["board"], original["board"])
        first.close()
        restored = self.session()
        with self.assertRaises(APIError):
            restored.new_game({})
        with self.assertRaises(APIError):
            restored.move({**move, "row": 2})
        completed = restored.move(move)
        self.assertEqual(completed["revision"], original["revision"] + 1)
        self.assertIsNone(completed["session"]["pending_move"])
        self.assertEqual(completed["board"][1][1], 2)

    def test_failed_durable_commit_does_not_acknowledge_or_mutate_board(self):
        first = self.session()
        original = first.snapshot()
        real_write = first._write_state
        def fail_commit(record):
            if record["pending"] is None and len(record["history"]) > 1:
                raise OSError("simulated disk failure")
            return real_write(record)
        move = {"row": 1, "col": 1, "revision": first.revision}
        with patch.object(first, "_write_state", side_effect=fail_commit), self.assertRaises(OSError):
            first.move(move)
        self.assertEqual(first.revision, original["revision"])
        self.assertEqual(first.snapshot()["board"], original["board"])
        self.assertEqual(json.loads(self.file.read_text(encoding="utf-8"))["pending"], move)
        self.assertEqual(first.move(move)["revision"], original["revision"] + 1)

    def test_changed_identity_and_inconsistent_history_are_rejected(self):
        first = self.session()
        first.close()
        with self.assertRaises(ValueError):
            self.session(identity={"run_id": "different"})
        record = json.loads(self.file.read_text(encoding="utf-8"))
        record["board"][0][0] = 1
        self.file.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.session()

    @staticmethod
    def windows_error(code):
        error = PermissionError(13, "simulated Windows replace failure")
        error.winerror = code
        return error

    def test_transient_windows_replace_reuses_one_fsynced_file_before_commit(self):
        import os
        first = self.session()
        original = first.snapshot()
        previous_bytes = self.file.read_bytes()
        replacements = []
        real_replace = os.replace
        real_fsync = os.fsync
        real_dump = json.dump

        def replace(source, destination):
            self.assertEqual(first.snapshot(), original)
            self.assertEqual(self.file.read_bytes(), previous_bytes)
            self.assertEqual(fsync.call_count, 1)
            replacements.append((Path(source), Path(source).read_bytes()))
            if len(replacements) <= 3:
                raise self.windows_error((5, 32, 33)[len(replacements) - 1])
            return real_replace(source, destination)

        with patch("play_unet.os.fsync", wraps=real_fsync) as fsync, \
             patch("play_unet.json.dump", wraps=real_dump) as dump, \
             patch("play_unet.os.replace", side_effect=replace), \
             patch("time.sleep") as sleep, patch("time.monotonic", return_value=0), \
             patch.object(first, "_analyze", wraps=first._analyze) as analysis:
            committed = first.new_game({"rows": 6, "cols": 7, "human_first": True})
        self.assertEqual(len(replacements), 4)
        self.assertEqual(len({path for path, _ in replacements}), 1)
        self.assertTrue(all(contents == replacements[0][1] for _, contents in replacements))
        self.assertEqual((fsync.call_count, dump.call_count, analysis.call_count), (1, 1, 1))
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(committed["revision"], original["revision"] + 1)
        self.assertEqual(json.loads(self.file.read_text(encoding="utf-8"))["board"], committed["board"])
        self.assertFalse(list(self.directory.glob("state.json.tmp-*")))

    def test_permanent_windows_replace_failure_preserves_durable_and_memory_state(self):
        import os
        first = self.session()
        original = first.snapshot()
        previous_bytes = self.file.read_bytes()
        replacements = []
        error = self.windows_error(5)
        real_fsync = os.fsync

        def fail(source, destination):
            replacements.append((Path(source), Path(source).read_bytes()))
            self.assertEqual(first.snapshot(), original)
            self.assertEqual(self.file.read_bytes(), previous_bytes)
            raise error

        with patch("play_unet.os.replace", side_effect=fail), \
             patch("play_unet.os.fsync", wraps=real_fsync) as fsync, \
             patch("time.sleep") as sleep, patch("time.monotonic", return_value=0), \
             self.assertRaises(PermissionError) as caught:
            first.new_game({"rows": 6, "cols": 7, "human_first": True})
        self.assertIs(caught.exception, error)
        self.assertEqual(len(replacements), 6)
        self.assertEqual(len({path for path, _ in replacements}), 1)
        self.assertTrue(all(contents == replacements[0][1] for _, contents in replacements))
        self.assertEqual(fsync.call_count, 1)
        self.assertEqual(sleep.call_count, 5)
        self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), .5)
        self.assertEqual(first.snapshot(), original)
        self.assertEqual(self.file.read_bytes(), previous_bytes)
        self.assertFalse(list(self.directory.glob("state.json.tmp-*")))
        first.close()
        restored = self.session()
        self.assertEqual(restored.snapshot(), original)

    def test_non_target_permission_and_io_errors_are_not_retried(self):
        first = self.session()
        original = first.snapshot()
        previous_bytes = self.file.read_bytes()
        for error in (PermissionError(13, "ordinary access denied"), OSError(28, "disk full"),
                      self.windows_error(1314)):
            with self.subTest(error=error), \
                 patch("play_unet.os.replace", side_effect=error) as replace, \
                 patch("time.sleep") as sleep, self.assertRaises(OSError) as caught:
                first.new_game({"rows": 6, "cols": 7, "human_first": True})
            self.assertIs(caught.exception, error)
            replace.assert_called_once()
            sleep.assert_not_called()
            self.assertEqual(first.snapshot(), original)
            self.assertEqual(self.file.read_bytes(), previous_bytes)
            self.assertFalse(list(self.directory.glob("state.json.tmp-*")))

    def test_retry_does_not_sleep_past_its_deadline(self):
        first = self.session()
        previous_bytes = self.file.read_bytes()
        error = self.windows_error(32)
        with patch("play_unet.os.replace", side_effect=error) as replace, \
             patch("time.monotonic", side_effect=[0, .49]), patch("time.sleep") as sleep, \
             self.assertRaises(PermissionError) as caught:
            first._write_state(first._state_record())
        self.assertIs(caught.exception, error)
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(self.file.read_bytes(), previous_bytes)
        self.assertFalse(list(self.directory.glob("state.json.tmp-*")))

    def test_overslept_retry_stops_before_attempting_another_replace(self):
        first = self.session()
        previous_bytes = self.file.read_bytes()
        error = self.windows_error(5)
        with patch("play_unet.os.replace", side_effect=error) as replace, \
             patch("time.monotonic", side_effect=[0, 0, .51]), patch("time.sleep") as sleep, \
             self.assertRaises(PermissionError) as caught:
            first._write_state(first._state_record())
        self.assertIs(caught.exception, error)
        replace.assert_called_once()
        sleep.assert_called_once()
        self.assertEqual(self.file.read_bytes(), previous_bytes)
        self.assertFalse(list(self.directory.glob("state.json.tmp-*")))

    def test_final_turn_replace_retry_does_not_repeat_ai_selection(self):
        import os
        selections = []
        def choose(board, side, analysis):
            selections.append((board.copy(), side))
            return dict(move=analysis["move"], engine="test")

        first = self.session(move_selector=choose)
        original = first.snapshot()
        move = {"row": 1, "col": 1, "revision": first.revision}
        commit_attempts = []
        real_replace = os.replace

        def replace(source, destination):
            contents = Path(source).read_bytes()
            record = json.loads(contents)
            if record["pending"] is None and len(record["history"]) > 1:
                self.assertEqual(first.revision, original["revision"])
                np.testing.assert_array_equal(first.board, original["board"])
                self.assertEqual(json.loads(self.file.read_text(encoding="utf-8"))["pending"], move)
                commit_attempts.append((Path(source), contents))
                if len(commit_attempts) == 1:
                    raise self.windows_error(33)
            return real_replace(source, destination)

        with patch("play_unet.os.replace", side_effect=replace) as replace_mock, \
             patch("time.sleep") as sleep:
            committed = first.move(move)
        self.assertEqual(len(selections), 1)
        self.assertEqual(replace_mock.call_count, 3)  # Intent once; final commit twice.
        self.assertEqual(len(commit_attempts), 2)
        self.assertEqual(commit_attempts[0], commit_attempts[1])
        sleep.assert_called_once()
        self.assertEqual(committed["revision"], original["revision"] + 1)
        self.assertIsNone(committed["session"]["pending_move"])
        self.assertIsNone(json.loads(self.file.read_text(encoding="utf-8"))["pending"])


if __name__ == "__main__":
    unittest.main()
