import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import web_match_runner as runner


def position(moves=(), ai_side=1, revision=1):
    board = np.zeros((16, 16), dtype=np.uint8)
    history = []
    for index, (row, col) in enumerate(moves):
        side = 1 + index % 2
        board[row, col] = side
        history.append(dict(side=side, row=row, col=col, by="ai" if side==ai_side else "human"))
    won = runner.winner(board)
    return dict(board=board.tolist(), history=history, ai_side=ai_side, human_side=3-ai_side,
                revision=revision, winner=won, finished=bool(won or not np.any(board==0)),
                turn=0 if won else 3-ai_side, move_count=len(history),
                session=dict(run_id="test", persisted=True, configuration_sha256="a", pending_move=None))


WIN = [(8, 8), (0, 0), (8, 9), (0, 2), (8, 10), (0, 4), (8, 11), (0, 6), (8, 12)]



class AtomicJSONTests(unittest.TestCase):
    @staticmethod
    def permission_error(winerror):
        error = PermissionError(13, "rename temporarily denied")
        if winerror is not None:
            error.winerror = winerror
        return error

    def test_transient_windows_read_lock_retries_same_complete_file(self):
        real_replace = runner.os.replace
        for winerror in (5, 32, 33):
            with self.subTest(winerror=winerror), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"run_status.json"
                original = b'{"revision":7,"status":"running"}'
                path.write_bytes(original)
                replacement = {"revision": 8, "status": "running", "note": "完整状态"}
                temporary_paths = []

                def replace_after_two_read_locks(source, target):
                    self.assertEqual(Path(target), path)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(json.loads(Path(source).read_text(encoding="utf-8")), replacement)
                    temporary_paths.append(Path(source))
                    if len(temporary_paths) < 3:
                        raise self.permission_error(winerror)
                    real_replace(source, target)

                with patch.object(runner.os, "replace", side_effect=replace_after_two_read_locks) as replace, \
                        patch.object(runner.os, "fsync", wraps=runner.os.fsync) as fsync, \
                        patch.object(runner.time, "monotonic", return_value=0), \
                        patch.object(runner.time, "sleep") as sleep:
                    runner.atomic_json(path, replacement)
                self.assertEqual(replace.call_count, 3)
                self.assertEqual(fsync.call_count, 1)
                self.assertEqual(sleep.call_args_list, [((.025,),), ((.05,),)])
                self.assertEqual(len(set(temporary_paths)), 1)
                self.assertEqual(path.read_text(encoding="utf-8"), runner.canonical_json(replacement))
                self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_persistent_lock_is_bounded_preserves_original_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"run_status.json"
            original = b'{ "revision": 7, "status": "running" }\n'
            path.write_bytes(original)
            error = self.permission_error(5)
            with patch.object(runner.os, "replace", side_effect=error) as replace, \
                    patch.object(runner.time, "monotonic", return_value=0), \
                    patch.object(runner.time, "sleep") as sleep:
                with self.assertRaises(PermissionError) as failure:
                    runner.atomic_json(path, {"revision": 8})
            self.assertIs(failure.exception, error)
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(sleep.call_count, 5)
            self.assertLess(sum(call.args[0] for call in sleep.call_args_list), .5)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_other_io_failures_are_never_retried(self):
        errors = [OSError(28, "disk full"), FileNotFoundError(2, "missing directory"),
                  self.permission_error(None), self.permission_error(87)]
        non_permission = OSError("not a permission failure")
        non_permission.winerror = 5
        errors.append(non_permission)
        for error in errors:
            with self.subTest(error=type(error).__name__, winerror=getattr(error, "winerror", None)), \
                    tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"run_status.json"
                original = b'{"revision":7}'
                path.write_bytes(original)
                with patch.object(runner.os, "replace", side_effect=error) as replace, \
                        patch.object(runner.time, "sleep") as sleep:
                    with self.assertRaises(OSError) as failure:
                        runner.atomic_json(path, {"revision": 8})
                self.assertIs(failure.exception, error)
                replace.assert_called_once()
                sleep.assert_not_called()
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_deadline_prevents_retry_that_would_exceed_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"run_status.json"
            original = b'{"revision":7}'
            path.write_bytes(original)
            error = self.permission_error(32)
            with patch.object(runner.os, "replace", side_effect=error) as replace, \
                    patch.object(runner.time, "monotonic", side_effect=[0, .49]), \
                    patch.object(runner.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    runner.atomic_json(path, {"revision": 8})
            replace.assert_called_once()
            sleep.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_overslept_deadline_does_not_attempt_another_rename(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"run_status.json"
            original = b'{"revision":7}'
            path.write_bytes(original)
            error = self.permission_error(33)
            with patch.object(runner.os, "replace", side_effect=error) as replace, \
                    patch.object(runner.time, "monotonic", side_effect=[0, .01, .6]), \
                    patch.object(runner.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    runner.atomic_json(path, {"revision": 8})
            replace.assert_called_once()
            sleep.assert_called_once_with(.025)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_serialization_failure_never_attempts_replace_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"run_status.json"
            original = b'{"revision":7}'
            path.write_bytes(original)
            with patch.object(runner.os, "replace") as replace, patch.object(runner.time, "sleep") as sleep:
                with self.assertRaises(ValueError):
                    runner.atomic_json(path, {"invalid": float("nan")})
            replace.assert_not_called()
            sleep.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])

class RunnerTests(unittest.TestCase):
    def test_journal_hash_chain_rejects_modified_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"events.jsonl"
            journal = runner.Journal(path)
            journal.append("one", {"winner": 1})
            journal.append("two", {"winner": 2})
            self.assertEqual(len(runner.Journal(path).events), 2)
            text = path.read_text(encoding="utf-8").replace('"winner":1', '"winner":2')
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.Journal(path)

    def test_trajectory_d4_color_and_actor_equivalence(self):
        first = position(WIN)["history"]
        transformed = [{**m, "row": m["col"], "col": 15-m["row"], "side": 3-m["side"]} for m in first]
        self.assertEqual(runner.trajectory_key(first, 1), runner.trajectory_key(transformed, 2))
        self.assertNotEqual(runner.trajectory_key(first, 1), runner.trajectory_key(first, 2))
        changed = copy.deepcopy(first)
        changed[1]["col"] = 1
        self.assertNotEqual(runner.trajectory_key(first, 1), runner.trajectory_key(changed, 1))

    def test_duplicate_never_adds_coverage_and_black_loss_is_retained(self):
        manifest = dict(formal=False, run_id="test", candidate_id="candidate")
        event = lambda side, result, key: {"kind": "game_completed", "data":
            dict(ai_side=side, outcome=result, canonical_trajectory=key)}
        events = [event(1, "win", "a"), event(1, "win", "a"), event(2, "draw", "b"),
                  event(1, "loss", "c"), event(1, "loss", "c")]
        status = runner.summarize(manifest, events)
        self.assertEqual(status["completed_attempts"], 5)
        self.assertEqual(status["ai_first_unique_games"], 2)
        self.assertEqual(status["duplicate_complete_games"], 2)
        self.assertEqual(status["black_losses"], 2)
        self.assertEqual(status["white_nonloss_rate"], 1)
        self.assertEqual(status["status"], "candidate_failed")
        self.assertFalse(status["counts_toward_formal_goal"])

    def test_goals_require_formal_and_white_strictly_above_half(self):
        manifest = dict(formal=True, run_id="test", candidate_id="candidate")
        events = [{"kind": "game_completed", "data": dict(ai_side=side,
                  outcome="win" if side==1 or i<500 else "loss", canonical_trajectory=f"{side}:{i}")}
                  for side in (1, 2) for i in range(1000)]
        self.assertFalse(runner.summarize(manifest, events)["goal_met"])
        events[-1]["data"]["outcome"] = "draw"
        self.assertTrue(runner.summarize(manifest, events)["goal_met"])
        self.assertFalse(runner.summarize({**manifest, "formal": False}, events)["goal_met"])

    def test_extra_white_wins_cannot_rescue_failed_first_thousand(self):
        manifest = dict(formal=True, run_id="test", candidate_id="candidate")
        events = [{"kind": "game_completed", "data": dict(ai_side=side,
                  outcome="win" if side==1 or i<500 else "loss", canonical_trajectory=f"{side}:{i}")}
                  for side in (1, 2) for i in range(1000)]
        for i in range(1000, 1100):
            events.append({"kind": "game_completed", "data": dict(
                ai_side=2, outcome="win", canonical_trajectory=f"2:{i}")})
        result = runner.summarize(manifest, events)
        self.assertEqual(result["white_nonloss_rate"], .5)
        self.assertEqual(result["ai_second_unique_games"], 1000)
        self.assertEqual(result["status"], "candidate_failed")
        self.assertFalse(result["goal_met"])
        self.assertIsNone(runner.next_ai_side(manifest, events, 2100))

    def test_full_color_quota_only_starts_other_color(self):
        manifest = dict(formal=True, run_id="test", candidate_id="candidate")
        def games(side):
            return [{"kind": "game_completed", "data": dict(ai_side=side, outcome="win",
                    canonical_trajectory=f"{side}:{i}")} for i in range(1000)]
        self.assertEqual(runner.next_ai_side(manifest, games(1), 1000), 2)
        self.assertEqual(runner.next_ai_side(manifest, games(2), 1001), 1)
        duplicated = games(1)[:999]
        duplicated.append(duplicated[-1])
        self.assertEqual(runner.next_ai_side(manifest, duplicated, 1000), 1)

    def test_failed_white_sample_remains_failed_across_runs(self):
        records = []
        for run_id, wins in (("early_success", 1000), ("failed", 500), ("later_success", 1000)):
            records.extend(dict(formal=True, run_id=run_id, ai_side=2,
                           outcome="win" if i<wins else "loss",
                           canonical_trajectory=f"{run_id}:{i}") for i in range(1000))
        result = runner.registered_candidate_failures(records)
        self.assertEqual(result["white_failed_runs"], ["failed"])
        # Later extra successes in the same failed run also cannot change its fixed sample.
        records.extend(dict(formal=True, run_id="failed", ai_side=2, outcome="win",
                       canonical_trajectory=f"extra:{i}") for i in range(100))
        self.assertEqual(runner.registered_candidate_failures(records)["white_failed_runs"], ["failed"])
        records.append(dict(formal=False, run_id="probe", ai_side=1, outcome="loss",
                            canonical_trajectory="probe_loss"))
        self.assertEqual(runner.registered_candidate_failures(records)["black_losses"], 1)

    def test_running_phase_does_not_override_failure_error_or_counts(self):
        manifest = dict(formal=True, run_id="test", candidate_id="candidate")
        paused = runner.summarize(manifest, [])
        running = runner.summarize(manifest, [], running=True)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(running["status"], "running")
        for key in ("completed_attempts", "black_losses", "white_nonloss_rate", "goal_met"):
            self.assertEqual(running[key], paused[key])
        self.assertEqual(runner.summarize(manifest, [], "I/O failed", running=True)["status"], "run_error")
        failed = [{"kind": "game_completed", "data": dict(ai_side=1, outcome="loss", canonical_trajectory="a")}]
        self.assertEqual(runner.summarize(manifest, failed, running=True)["status"], "candidate_failed")
        white_failed = [{"kind": "game_completed", "data": dict(
            ai_side=2, outcome="loss", canonical_trajectory=str(i))} for i in range(1000)]
        self.assertEqual(runner.summarize(manifest, white_failed, running=True)["status"], "candidate_failed")

    def test_terminal_must_be_actual_five_not_truncation(self):
        live = position([(8, 8)])
        runner.verify_state(live)
        live["finished"] = True
        with self.assertRaises(ValueError):
            runner.verify_state(live)
        terminal = position(WIN)
        self.assertEqual(runner.verify_state(terminal)[1:], (1, True))
        terminal["history"].append(dict(side=2, row=15, col=15, by="human"))
        terminal["board"][15][15] = 2
        terminal["move_count"] += 1
        with self.assertRaises(ValueError):
            runner.verify_state(terminal)

    def test_board_history_turn_and_forbidden_rejected(self):
        board = position([(8, 8)])
        board["board"][1][1] = 3
        with self.assertRaises(ValueError):
            runner.verify_state(board)
        board = position([(8, 8)])
        board["history"][0]["by"] = "human"
        with self.assertRaises(ValueError):
            runner.verify_state(board)
        with self.assertRaises(ValueError):
            runner.verify_state(position([(8, 8), (1, 1)]))

    def test_exact_new_game_and_atomic_human_ai_transition(self):
        before = position([(8, 8)], revision=5)
        after = position([(8, 8), (1, 1), (8, 9)], revision=6)
        runner.verify_transition(before, after, "move", [1, 1], 1)
        with self.assertRaises(ValueError):
            runner.verify_transition(before, after, "move", [1, 2], 1)
        runner.verify_transition(before, position((), ai_side=2, revision=6), "new", ai_side=2)
        with self.assertRaises(ValueError):
            runner.verify_transition(before, position([(7, 7)], revision=6), "new", ai_side=1)

    def test_restored_committed_click_is_not_replayed(self):
        before = position([(8, 8)], revision=5)
        after = position([(8, 8), (1, 1), (8, 9)], revision=6)
        snap = {"state": after, "dom": {"error": ""}}
        class Fake:
            def wait_state(self): return snap
            def find_receipt(self, snapshot): return {"body": after, "method": "GET"}
            def click_cell(self, *move): raise AssertionError("Committed move must never be replayed")
        intent = dict(kind="move", game_id="g", before=before, move=[1, 1], ai_side=1, intent_seq=2)
        with patch.object(runner, "save_evidence", return_value=snap) as save:
            result = runner.perform_action(".", None, Fake(), "g", {"state": before}, "move",
                                           ai_side=1, existing_intent=intent)
            self.assertIs(result, snap)
            self.assertTrue(save.call_args.kwargs["recovered"])

    def test_terminal_result_recovery_keeps_same_result_and_registry(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            manifest = dict(run_id="run", candidate_id="candidate", formal=False,
                            registry=str(run/"registry.jsonl"))
            journal = runner.Journal(run/"events.jsonl")
            state = position(WIN)
            result = runner.finish_game(run, manifest, journal, "game_1", state)
            self.assertEqual(result["outcome"], "win")
            self.assertEqual(len(runner.synchronize_registry(run, manifest, journal)), 1)
            saved_result = run/result["result_file"]
            before = saved_result.read_bytes()
            # Simulate crash after result fsync but before its completion event.
            other = runner.Journal(run/"recovered.jsonl")
            runner.finish_game(run, manifest, other, "game_1", state)
            self.assertEqual(saved_result.read_bytes(), before)
            self.assertEqual(len(runner.synchronize_registry(run, manifest, other)), 1)

    def test_complete_runner_loop_preserves_terminal_result(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            manifest = dict(formal=False, run_id="run", candidate_id="candidate", seed=7,
                            opponent_options={}, registry=str(run/"registry.jsonl"))
            class FakeRuntime:
                def __init__(self, *args):
                    assert json.loads((run/"run_status.json").read_text(encoding="utf-8"))["status"] == "running"
                    self.current = {"state": position([(8, 8)])}
                    self.opponent = self
                def wait_state(self): return self.current
                def request(self, request):
                    assert json.loads((run/"run_status.json").read_text(encoding="utf-8"))["status"] == "running"
                    index = request["context"]["ply"]
                    return {"move": list(WIN[index]), "reason": "fixture"}
                def close(self, **kwargs): pass
            def perform(run, journal, runtime, game_id, before, kind, ai_side=None,
                        choice=None, existing_intent=None):
                previous = before["state"]
                moves = [(m["row"],m["col"]) for m in previous["history"]]
                if kind=="new":
                    moves = [(8,8)]
                else:
                    moves.extend([tuple(choice["move"]), WIN[len(moves)+1]])
                runtime.current = {"state": position(moves, revision=previous["revision"]+1)}
                return runtime.current
            with patch.object(runner, "Runtime", FakeRuntime), patch.object(runner, "perform_action", side_effect=perform):
                self.assertEqual(runner.run_games(run, manifest, 1), 0)
            status = json.loads((run/"run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["completed_attempts"], 1)
            self.assertEqual(status["status"], "paused")
            self.assertEqual(status["outcomes"]["1"]["win"], 1)
            result = json.loads((run/"games/game_000001/result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["plies"], 9)
            self.assertEqual(result["termination"], "five_or_more")

    def test_protected_game_ports_are_never_contacted(self):
        for port in (8765, 8766):
            with self.assertRaises(ValueError):
                runner.listening(port)

    def test_guarded_selector_shares_total_budget_and_preserves_guard_status(self):
        from play_unet import build_move_selector
        prior = np.full((16, 16), 1/256)
        result = dict(move=(0, 0), reason="Budget-limited guard", proven_value=None,
                      guard_changed=True, completed_depth=0, value_evaluations=0)
        with patch("search_guard.select_guarded_move", return_value=result) as search, \
             patch("play_unet.describe_ai_move", return_value="structural development"):
            select = build_move_selector(object(), engine="guarded", search_seconds=.5,
                                         search_nodes=50000, guard_probe_nodes=2000)
            selected = select(np.zeros((16, 16), dtype=np.uint8), 1, {"raw_play_policy": prior})
            self.assertEqual(search.call_args.kwargs["max_nodes"], 50000)
            self.assertEqual(search.call_args.kwargs["time_limit"], .5)
            self.assertEqual(search.call_args.kwargs["forcing_depth"], 32)
            self.assertEqual(search.call_args.kwargs["probe_max_nodes"], 2000)
            self.assertEqual(search.call_args.kwargs["threat_time_limit"], .2)
            self.assertEqual(search.call_args.kwargs["threat_max_nodes"], 20000)
            self.assertEqual(search.call_args.kwargs["threat_width"], 16)
            self.assertEqual(search.call_args.kwargs["threat_quiet_plies"], 2)
            self.assertEqual(search.call_args.kwargs["threat_total_plies"], 64)
            self.assertEqual(search.call_args.kwargs["attack_time_limit"], .15)
            self.assertEqual(search.call_args.kwargs["attack_max_nodes"], 20000)
            self.assertNotIn("forcing_seconds", search.call_args.kwargs)
            self.assertNotIn("value_evaluator", search.call_args.kwargs)
            self.assertTrue(selected["reason"].startswith("Budget-limited guard"))

    def test_quiet_guard_flags_rejected_by_other_engines(self):
        import io
        from contextlib import redirect_stderr
        import play_unet
        for option, value in (("--guard-threat-seconds", ".2"),
                              ("--guard-threat-nodes", "20000"),
                              ("--guard-threat-width", "16"),
                              ("--guard-threat-quiet-plies", "2"),
                              ("--guard-threat-total-plies", "64"),
                              ("--guard-attack-seconds", ".15"),
                              ("--guard-attack-nodes", "20000")):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    play_unet.main(["--engine", "native", option, value])
                self.assertEqual(caught.exception.code, 2)
                with patch.object(runner, "freeze_run", side_effect=AssertionError("cannot freeze invalid config")):
                    with self.assertRaises(SystemExit) as caught:
                        runner.main(["--run-dir", "unused", "--engine", "python", option, value])
                self.assertEqual(caught.exception.code, 2)

    def test_default_guard_cli_reaches_selector_and_http_configuration(self):
        import io
        from contextlib import redirect_stdout
        import play_unet
        with patch("unet_pipeline.load_model", return_value=(None, {"trained": True})), \
             patch("native_search.native_fingerprint", return_value={}), \
             patch.object(play_unet, "build_move_selector") as selector, \
             patch.object(play_unet, "GameSession") as session, \
             patch.object(play_unet, "LocalGameServer"), redirect_stdout(io.StringIO()):
            self.assertEqual(play_unet.main(["--models", "test-models-not-loaded", "--engine", "guarded"]), 0)
        for field, expected in (("guard_threat_quiet_plies", 2), ("guard_threat_total_plies", 64),
                                ("guard_attack_seconds", .15), ("guard_attack_nodes", 20000)):
            self.assertEqual(selector.call_args.kwargs[field], expected)
            self.assertEqual(session.call_args.kwargs["search_configuration"][field], expected)

    def test_default_guard_runner_cli_passes_new_configuration_to_freeze(self):
        class ConfigurationCaptured(Exception):
            pass
        with patch.object(runner, "freeze_run", side_effect=ConfigurationCaptured) as freeze:
            with self.assertRaises(ConfigurationCaptured):
                runner.main(["--run-dir", "unused-configuration-test", "--engine", "guarded"])
        args = freeze.call_args.args[0]
        self.assertEqual(args.guard_threat_quiet_plies, 2)
        self.assertEqual(args.guard_threat_total_plies, 64)
        self.assertEqual(args.guard_attack_seconds, .15)
        self.assertEqual(args.guard_attack_nodes, 20000)

    def test_guard_depth_cli_ranges_match_prover_and_reject_out_of_range(self):
        import io
        from contextlib import redirect_stderr
        import play_unet
        for flag, value in (("--guard-threat-quiet-plies", "-1"),
                            ("--guard-threat-quiet-plies", "33"),
                            ("--guard-threat-total-plies", "0"),
                            ("--guard-threat-total-plies", "257")):
            with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    play_unet.main(["--engine", "guarded", flag, value])
                self.assertEqual(caught.exception.code, 2)
                with patch.object(runner, "freeze_run", side_effect=AssertionError("must reject before freeze")):
                    with self.assertRaises(SystemExit) as caught:
                        runner.main(["--run-dir", "unused", "--engine", "guarded", flag, value])
                self.assertEqual(caught.exception.code, 2)

    def test_guard_depth_cli_accepts_both_range_endpoints(self):
        import io
        from contextlib import redirect_stdout
        import play_unet
        class ConfigurationCaptured(Exception):
            pass
        for quiet, total in ((0, 1), (32, 256)):
            flags = ["--engine", "guarded", "--guard-threat-quiet-plies", str(quiet),
                     "--guard-threat-total-plies", str(total)]
            with self.subTest(quiet=quiet, total=total), \
                 patch("unet_pipeline.load_model", return_value=(None, {"trained": True})), \
                 patch("native_search.native_fingerprint", return_value={}), \
                 patch.object(play_unet, "build_move_selector") as selector, \
                 patch.object(play_unet, "GameSession"), \
                 patch.object(play_unet, "LocalGameServer"), redirect_stdout(io.StringIO()):
                self.assertEqual(play_unet.main(["--models", "test-models-not-loaded"]+flags), 0)
                self.assertEqual(selector.call_args.kwargs["guard_threat_quiet_plies"], quiet)
                self.assertEqual(selector.call_args.kwargs["guard_threat_total_plies"], total)
                with patch.object(runner, "freeze_run", side_effect=ConfigurationCaptured) as freeze:
                    with self.assertRaises(ConfigurationCaptured):
                        runner.main(["--run-dir", "unused"]+flags)
                self.assertEqual(freeze.call_args.args[0].guard_threat_quiet_plies, quiet)
                self.assertEqual(freeze.call_args.args[0].guard_threat_total_plies, total)

    def test_quiet_proof_evidence_survives_game_response(self):
        from play_unet import GameSession
        from tests.test_play_persistence import analyze
        evidence = {"move": [0, 0], "opponent_result": {"proven_value": None},
                    "opponent_threat_result": {"proven_value": None, "defenses_checked": 225},
                    "rejection_source": None}
        selected = dict(move=(0, 0), reason="bounded quiet probe", nodes=17, completed_depth=0,
                        guard_probes=[evidence], selected_reply=evidence,
                        threat_time_limit=.2, threat_max_nodes=20000, threat_width=16,
                        threat_quiet_plies=2, threat_total_plies=64,
                        attack_time_limit=.15, attack_max_nodes=20000,
                        attack_result={"proven_value": None, "nodes": 13, "principal_variation": []})
        session = GameSession(None, None, analyzer=analyze,
                              move_selector=lambda *args: selected, engine="guarded",
                              search_configuration={"guard_threat_seconds": .2})
        state = session.move({"row": 1, "col": 1, "revision": session.revision})
        self.assertEqual(state["last_ai_search"]["guard_probes"], [evidence])
        self.assertEqual(state["last_ai_search"]["threat_time_limit"], .2)
        self.assertEqual(state["last_ai_search"]["attack_result"], selected["attack_result"])
        for field in ("threat_quiet_plies", "threat_total_plies", "attack_time_limit", "attack_max_nodes"):
            self.assertEqual(state["last_ai_search"][field], selected[field])
        self.assertEqual(state["search_configuration"]["guard_threat_seconds"], .2)

    def test_guard_rejects_independent_forcing_budget_and_native_size_limit(self):
        from play_unet import build_move_selector, GameSession
        with self.assertRaises(ValueError):
            build_move_selector(engine="guarded", forcing_seconds=.1)
        with self.assertRaises(ValueError):
            GameSession(None, None, engine="native", max_size=65, auto_start=False)
        with self.assertRaises(ValueError):
            GameSession(None, None, engine="guarded", max_size=65, auto_start=False)

    def test_native_selector_uses_combined_prior_without_neural_leaf(self):
        from play_unet import build_move_selector
        raw = np.full((16, 16), 1/256)
        combined = raw.copy()
        combined[0, 0] = .3
        result = dict(move=(0, 0), reason="search", value_evaluations=0)
        with patch("native_search.select_native_move", return_value=result) as search, \
             patch("play_unet.describe_ai_move", return_value="search"), \
             patch("global_inference.evaluate_global_value", side_effect=AssertionError("must not call")):
            select = build_move_selector(object(), engine="native", search_depth=9)
            select(np.zeros((16, 16), dtype=np.uint8), 1,
                   {"raw_play_policy": raw, "combined_policy": combined})
            np.testing.assert_array_equal(search.call_args.args[2], combined)
            self.assertNotIn("value_evaluator", search.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
