"""Offline artifact tests; no browser, service, model or search is started."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from tools.browser.match_evidence import ROOT, verify_game_evidence
from tools.browser.match_verify import file_hash, verify_terminal_game
from tools.browser.search_budget import BUDGET_MODULE, V2_MODULE, V2_BUDGET
from tests.browser.test_match_verify import raw, state, BLACK_WIN, WHITE_WIN

# A valid one-pixel PNG keeps the fixture independent of image libraries.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415408d763f8cfc0f01f00050001ff89993d1d0000000049454e44ae426082"
)
OPTIONS = dict(time_limit=1.5, max_nodes=150000, depth=9, candidate_width=24,
               forcing_seconds=.15, forcing_nodes=20000, forcing_depth=32)


class Fixture:
    def __init__(self, run, history=BLACK_WIN, ai_side=2):
        self.run = Path(run)
        self.history, self.ai_side = list(history), ai_side
        asset = self.run / "frozen/web/assets.json"
        asset.parent.mkdir(parents=True)
        asset.write_text('{"version":1}', encoding="utf-8")
        self.manifest = dict(run_id="run-test", candidate_id="candidate-test", seconds=1,
                             web_hashes={"assets.json": file_hash(asset)}, opponent_options=OPTIONS)
        self.events = []
        self.position(0)
        opponent = []
        for ply, point in enumerate(self.history):
            before = state(self.history[:ply], ai_side)
            side = 1 + ply % 2
            move = list(divmod(point, 16))
            if side != ai_side:
                self.emit("input", dict(type="click", trusted=True, point=str(point),
                                       history=self.history[:ply], target="DIV"))
                choice = dict(move=move, reason="bounded search", search=dict(move=move, nodes=123),
                              challenger="full_board_native_vcf_challenger_v1",
                              root_order_seed="a" * 64, options=copy.deepcopy(OPTIONS),
                              neural_value_used=False)
                opponent.append(dict(before=before, choice=choice))
            else:
                request = dict(type="analyze", id=100+ply, n=16, side=side,
                               board=raw(self.history[:ply], ai_side)["board"], seconds=1)
                self.emit("analysis_requested", request)
                response = dict(type="result", id=request["id"], seconds=1,
                                search=dict(move=point, nodes=123, value=None),
                                elapsedMs=1200, inferenceMs=250, overrunMs=200,
                                valueUsedInSearch=False)
                self.emit("analysis_result", dict(request=copy.deepcopy(request), result=response))
            self.position(ply+1)
        self.result = verify_terminal_game(state(self.history, ai_side))
        self.result.update(game_id="game_000001", evidence=[], opponent_moves=opponent)
        self.payloads = []
        self.add_payload(self.events)

    def emit(self, kind, data):
        self.events.append(dict(seq=len(self.events)+1, at=10.0*(len(self.events)+1), kind=kind, data=data))

    def position(self, ply):
        snapshot = raw(self.history[:ply], self.ai_side,
                       storageError=None, storageBlocked=False, pendingCommit=None)
        self.emit("position", dict(state=snapshot,
                                  visible_cells=[dict(point=p, cell=v) for p, v in enumerate(snapshot["board"])],
                                  saved=json.dumps({key: snapshot[key] for key in
                                                    ("n", "board", "history", "human", "seconds", "started")})))

    def add_payload(self, events, page="page-test"):
        number = len(self.payloads)
        folder = self.run / "games/game_000001"
        folder.mkdir(parents=True, exist_ok=True)
        screenshot = folder / f"capture{number}.png"
        screenshot.write_bytes(PNG)
        normalized = state(self.history, self.ai_side)
        snapshot = raw(self.history, self.ai_side)
        payload = dict(run_id=self.manifest["run_id"], candidate_id=self.manifest["candidate_id"],
                       game_id=self.result["game_id"], stage="terminal",
                       raw=snapshot, state=normalized, saved=copy.deepcopy(snapshot),
                       audit=dict(page_id=page, events=copy.deepcopy(events)),
                       screenshot=screenshot.relative_to(self.run).as_posix())
        path = folder / f"capture{number}.json.gz"
        self.result["evidence"].append(dict(file=path.relative_to(self.run).as_posix(),
                                           sha256="", screenshot=payload["screenshot"],
                                           screenshot_sha256=file_hash(screenshot)))
        self.payloads.append(payload)
        self.write(number)

    def write(self, index=0):
        reference = self.result["evidence"][index]
        path = self.run / reference["file"]
        path.write_bytes(gzip.compress(json.dumps(self.payloads[index]).encode("utf-8"), mtime=0))
        reference["sha256"] = file_hash(path)

    def verify(self):
        return verify_game_evidence(self.run, self.manifest, self.result)

    def mutate_event(self, kind, change):
        event = next(event for event in self.payloads[0]["audit"]["events"] if event["kind"] == kind)
        change(event)
        self.write()


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def enable_v2_budget(self):
        module = self.f.run / "frozen/web" / BUDGET_MODULE
        module.write_bytes(V2_MODULE.encode("utf-8"))
        self.f.manifest["web_hashes"][BUDGET_MODULE] = file_hash(module)
        self.f.manifest["search_budget"] = dict(V2_BUDGET)
        for event in self.f.payloads[0]["audit"]["events"]:
            if event["kind"] == "analysis_result":
                event["data"]["result"]["search"].update(
                    nodes=750000, nodeLimit=1000000, budgetVersion=2)
        self.f.write()

    def test_versioned_guard_accepts_more_nodes_only_for_its_frozen_candidate(self):
        self.assertEqual((ROOT / "web/browser" / BUDGET_MODULE).read_bytes(), V2_MODULE.encode("utf-8"))
        self.enable_v2_budget()
        audit = self.f.verify()
        self.assertEqual(audit["ai_budget"]["max_nodes"], 1000000)
        self.assertEqual(audit["ai_budget"]["total_nodes"], 4 * 750000)
        self.assertEqual(audit["ai_budget"]["seconds"], 1)

    def test_legacy_receipt_cannot_claim_the_new_node_guard(self):
        self.f.mutate_event("analysis_result", lambda e: e["data"]["result"]["search"].update(
            nodes=150001, nodeLimit=1000000, budgetVersion=2))
        with self.assertRaisesRegex(ValueError, "AI nodes"):
            self.f.verify()

    def test_new_budget_declaration_does_not_relax_missing_legacy_module(self):
        self.f.manifest["search_budget"] = dict(V2_BUDGET)
        with self.assertRaisesRegex(ValueError, "lacks its frozen module"):
            self.f.verify()

    def test_v2_module_requires_an_exact_frozen_declaration(self):
        self.enable_v2_budget()
        for field, value in (("version", True), ("nodes_per_second", 2000000),
                             ("minimum_nodes", 15000.0), ("sha256", "0" * 64)):
            self.f.manifest["search_budget"] = dict(V2_BUDGET, **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "declaration differs"):
                self.f.verify()
        self.f.manifest.pop("search_budget")
        with self.assertRaisesRegex(ValueError, "declaration differs"):
            self.f.verify()

    def test_unknown_module_is_rejected_even_after_rehashing_manifest(self):
        self.enable_v2_budget()
        module = self.f.run / "frozen/web" / BUDGET_MODULE
        module.write_bytes(V2_MODULE.replace("1000000", "9000000").encode("utf-8"))
        self.f.manifest["web_hashes"][BUDGET_MODULE] = file_hash(module)
        with self.assertRaisesRegex(ValueError, "Unknown or changed"):
            self.f.verify()

    def test_new_guard_cannot_be_increased_or_omitted_by_worker_receipt(self):
        self.enable_v2_budget()
        original = copy.deepcopy(self.f.payloads[0])
        for field, value in (("nodes", 1000001), ("nodes", True),
                             ("nodeLimit", 2000000), ("budgetVersion", 1),
                             ("nodeLimit", None), ("budgetVersion", None)):
            self.f.payloads[0] = copy.deepcopy(original)
            def change(event):
                search = event["data"]["result"]["search"]
                if value is None:
                    search.pop(field)
                else:
                    search[field] = value
            self.f.mutate_event("analysis_result", change)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.f.verify()

    def test_explicit_square_columns_keep_the_legacy_audit_identical(self):
        expected = self.f.verify()
        payload = self.f.payloads[0]
        payload["raw"]["cols"] = 16
        payload["saved"]["cols"] = 16
        for event in payload["audit"]["events"]:
            data = event["data"]
            if event["kind"] == "position":
                data["state"]["cols"] = 16
                saved = json.loads(data["saved"])
                saved["cols"] = 16
                data["saved"] = json.dumps(saved)
            elif event["kind"] == "analysis_requested":
                data["cols"] = 16
            elif event["kind"] == "analysis_result":
                data["request"]["cols"] = 16
        self.f.write()
        self.assertEqual(self.f.verify(), expected)
        # Migration can mix an old saved record with a new visible snapshot.
        del payload["saved"]["cols"]
        for event in payload["audit"]["events"]:
            if event["kind"] == "position":
                saved = json.loads(event["data"]["saved"])
                saved.pop("cols", None)
                event["data"]["saved"] = json.dumps(saved)
        self.f.write()
        self.assertEqual(self.f.verify(), expected)

    def test_incompatible_columns_cannot_hide_in_any_evidence_layer(self):
        original = copy.deepcopy(self.f.payloads[0])
        for target in ("raw", "payload_saved", "position", "position_saved", "request"):
            for cols in (12, True, 16.0, None):
                self.f.payloads[0] = copy.deepcopy(original)
                payload = self.f.payloads[0]
                if target == "raw":
                    payload["raw"]["cols"] = cols
                elif target == "payload_saved":
                    payload["saved"]["cols"] = cols
                elif target in ("position", "position_saved"):
                    data = next(e["data"] for e in payload["audit"]["events"] if e["kind"] == "position")
                    if target == "position":
                        data["state"]["cols"] = cols
                    else:
                        saved = json.loads(data["saved"])
                        saved["cols"] = cols
                        data["saved"] = json.dumps(saved)
                else:
                    # Keep request/result pairing consistent so the geometry
                    # validation itself must reject the forged request.
                    for event in payload["audit"]["events"]:
                        if event["kind"] == "analysis_requested":
                            event["data"]["cols"] = cols
                        elif event["kind"] == "analysis_result":
                            event["data"]["request"]["cols"] = cols
                self.f.write()
                with self.subTest(target=target, cols=cols), self.assertRaises(ValueError):
                    self.f.verify()

    def test_complete_replay_and_audit_are_deterministic_and_read_only(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[2])
        before = copy.deepcopy((self.f.manifest, self.f.result))
        hashes = {str(path): file_hash(path) for path in self.f.run.rglob("*") if path.is_file()}
        result = self.f.verify()
        self.assertEqual(result, self.f.verify())
        self.assertEqual((result["winner"], result["outcome"], result["plies"]), (1, "loss", 9))
        self.assertEqual((result["verified_prefixes"], result["human_moves"], result["ai_moves"]), (10, 5, 4))
        self.assertEqual(result["ai_budget"]["total_nodes"], 492)
        self.assertEqual(result["ai_budget"]["total_overrun_ms"], 800)
        self.assertFalse(result["complete_branch_certificates_verified"])
        self.assertEqual(before, (self.f.manifest, self.f.result))
        self.assertEqual(hashes, {str(path): file_hash(path) for path in self.f.run.rglob("*") if path.is_file()})

    def test_black_ai_white_human_and_either_winner(self):
        for history in (BLACK_WIN, WHITE_WIN):
            for ai_side in (1, 2):
                with self.subTest(ai_side=ai_side, winner=1 + (history == WHITE_WIN)), tempfile.TemporaryDirectory() as folder:
                    f = Fixture(folder, history, ai_side)
                    checked = f.verify()
                    self.assertEqual(checked["outcome"], "win" if checked["winner"] == ai_side else "loss")

    def test_human_click_must_be_trusted_matching_point_and_complete_prefix(self):
        original = copy.deepcopy(self.f.payloads[0])
        for field, value in (("trusted", False), ("trusted", 1), ("type", "keydown"),
                             ("point", "1"), ("point", True), ("history", [False])):
            self.f.payloads[0] = copy.deepcopy(original)
            self.f.mutate_event("input", lambda e: e["data"].__setitem__(field, value))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.f.verify()

    def test_every_actual_human_move_needs_matching_searched_choice(self):
        records = copy.deepcopy(self.f.result["opponent_moves"])
        for change in ("missing", "duplicate", "different_move", "different_search", "different_actor",
                       "different_budget", "neural_value", "wrong_game"):
            self.f.result["opponent_moves"] = copy.deepcopy(records)
            items = self.f.result["opponent_moves"]
            if change == "missing":
                items.pop()
            elif change == "duplicate":
                items.append(copy.deepcopy(items[0]))
            elif change == "different_move":
                items[0]["choice"]["move"] = [0, 1]
            elif change == "different_search":
                items[0]["choice"]["search"]["move"] = [0, 1]
            elif change == "different_actor":
                items[0]["before"] = state([], 1)
            elif change == "different_budget":
                items[0]["choice"]["options"]["time_limit"] = 10
            elif change == "neural_value":
                items[0]["choice"]["neural_value_used"] = True
            else:
                items[0]["game_id"] = "another-game"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.f.verify()

    def test_ai_result_request_and_selected_move_must_match_real_preceding_board(self):
        original = copy.deepcopy(self.f.payloads[0])
        for field, value in (("board", [0]*256), ("side", 1), ("seconds", 2), ("id", True)):
            self.f.payloads[0] = copy.deepcopy(original)
            self.f.mutate_event("analysis_result", lambda e: e["data"]["request"].__setitem__(field, value))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.f.verify()
        self.f.payloads[0] = copy.deepcopy(original)
        self.f.mutate_event("analysis_result", lambda e: e["data"]["result"]["search"].__setitem__("move", 100))
        with self.assertRaises(ValueError):
            self.f.verify()

    def test_ai_node_budget_and_finite_time_accounting_are_enforced(self):
        original = copy.deepcopy(self.f.payloads[0])
        cases = [("nodes", 150001), ("nodes", True), ("nodes", -1),
                 ("elapsedMs", float("nan")), ("elapsedMs", float("inf")),
                 ("inferenceMs", 1201), ("inferenceMs", -1), ("overrunMs", 0),
                 ("overrunMs", True), ("valueUsedInSearch", True), ("valueUsedInSearch", 0),
                 ("id", 999), ("seconds", 2)]
        for field, value in cases:
            self.f.payloads[0] = copy.deepcopy(original)
            def mutate(event):
                response = event["data"]["result"]
                (response["search"] if field == "nodes" else response)[field] = value
            self.f.mutate_event("analysis_result", mutate)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.f.verify()

    def test_ai_result_needs_observed_earlier_request_and_cannot_be_after_move(self):
        original = copy.deepcopy(self.f.payloads[0])
        self.f.mutate_event("analysis_requested", lambda e: e.__setitem__("kind", "other"))
        with self.assertRaisesRegex(ValueError, "earlier worker request"):
            self.f.verify()
        self.f.payloads[0] = copy.deepcopy(original)
        events = self.f.payloads[0]["audit"]["events"]
        index = next(i for i, e in enumerate(events) if e["kind"] == "analysis_result")
        events[index], events[index+1] = events[index+1], events[index]
        for i, event in enumerate(events):
            event.update(seq=i+1, at=float(i+1))
        self.f.write()
        with self.assertRaises(ValueError):
            self.f.verify()

    def test_repeated_same_request_does_not_erase_the_original_result_provenance(self):
        events = self.f.payloads[0]["audit"]["events"]
        result_index = next(i for i, event in enumerate(events) if event["kind"] == "analysis_result")
        request = copy.deepcopy(events[result_index-1])
        events.insert(result_index+1, request)
        for index, event in enumerate(events):
            event.update(seq=index+1, at=float(index+1))
        self.f.write()
        self.assertTrue(self.f.verify()["verified"])

    def test_small_subbudget_elapsed_has_zero_overrun(self):
        for event in self.f.payloads[0]["audit"]["events"]:
            if event["kind"] == "analysis_result":
                event["data"]["result"].update(elapsedMs=15.5, inferenceMs=10.25, overrunMs=0)
        self.f.write()
        result = self.f.verify()
        self.assertEqual(result["ai_budget"]["moves_over_budget"], 0)
        self.assertEqual(result["ai_budget"]["total_search_ms"], 21)

    def test_sequence_metadata_and_noninteger_request_numbers_are_strict(self):
        original = copy.deepcopy(self.f.payloads[0])
        for field, value in (("seq", True), ("seq", 0), ("at", False), ("at", -1),
                             ("at", float("inf")), ("kind", ""), ("data", [])):
            self.f.payloads[0] = copy.deepcopy(original)
            self.f.payloads[0]["audit"]["events"][0][field] = value
            self.f.write()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.f.verify()
        self.f.payloads[0] = copy.deepcopy(original)
        self.f.mutate_event("analysis_requested", lambda e: e["data"].__setitem__("side", True))
        with self.assertRaises(ValueError):
            self.f.verify()

    def test_all_prefixes_need_dom_and_saved_board_not_just_final_screenshot(self):
        original = copy.deepcopy(self.f.payloads[0])
        events = self.f.payloads[0]["audit"]["events"]
        position = next(e for e in events if e["kind"] == "position" and len(e["data"]["state"]["history"]) == 1)
        position["kind"] = "other"
        self.f.write()
        with self.assertRaisesRegex(ValueError, "prefixes"):
            self.f.verify()
        for field in ("visible_cells", "saved"):
            self.f.payloads[0] = copy.deepcopy(original)
            self.f.mutate_event("position", lambda e: e["data"].pop(field))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.f.verify()

    def test_dom_and_local_storage_tampering_and_boolean_cells_are_rejected(self):
        original = copy.deepcopy(self.f.payloads[0])
        for change in ("stone", "duplicate", "boolean", "saved_board", "saved_history", "saved_none",
                       "saved_identity", "blocked", "pending"):
            self.f.payloads[0] = copy.deepcopy(original)
            def mutate(event):
                data = event["data"]
                if change == "stone":
                    data["visible_cells"][0]["cell"] = 1
                elif change == "duplicate":
                    data["visible_cells"][1]["point"] = 0
                elif change == "boolean":
                    data["visible_cells"][0]["cell"] = False
                elif change.startswith("saved"):
                    saved = json.loads(data["saved"])
                    if change == "saved_board":
                        saved["board"][0] = 1
                    elif change == "saved_history":
                        saved["history"] = [False]
                    elif change == "saved_identity":
                        saved["human"] = 2
                    data["saved"] = None if change == "saved_none" else json.dumps(saved)
                elif change == "blocked":
                    data["state"]["storageBlocked"] = True
                else:
                    data["state"]["pendingCommit"] = {}
            self.f.mutate_event("position", mutate)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.f.verify()

    def test_file_png_hashes_and_frozen_assets_are_checked(self):
        reference = self.f.result["evidence"][0]
        original = copy.deepcopy(reference)
        for field in ("sha256", "screenshot_sha256"):
            reference.update(original)
            reference[field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.f.verify()
        reference.update(original)
        asset = self.f.run / "frozen/web/assets.json"
        asset.write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.f.verify()

    def test_missing_files_invalid_png_and_paths_outside_run_are_rejected(self):
        reference = self.f.result["evidence"][0]
        original = copy.deepcopy(reference)
        for value in ("missing.json.gz", "../outside.json.gz"):
            reference["file"] = value
            with self.assertRaises(ValueError):
                self.f.verify()
        reference.update(original)
        png = self.f.run / reference["screenshot"]
        png.write_bytes(b"not a png")
        reference["screenshot_sha256"] = file_hash(png)
        with self.assertRaisesRegex(ValueError, "not a PNG"):
            self.f.verify()

    def test_payload_raw_normalized_and_terminal_result_identity_are_revalidated(self):
        original = copy.deepcopy(self.f.payloads[0])
        for field, value in (("run_id", "other"), ("candidate_id", "other"), ("game_id", "other"),
                             ("stage", ""), ("screenshot", "missing.png")):
            self.f.payloads[0] = copy.deepcopy(original)
            self.f.payloads[0][field] = value
            self.f.write()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.f.verify()
        for change in ("raw_board", "state", "saved", "storage", "outcome", "trajectory"):
            self.f.payloads[0] = copy.deepcopy(original)
            if change == "raw_board":
                self.f.payloads[0]["raw"]["board"][0] = 0
            elif change == "state":
                self.f.payloads[0]["state"]["history"][0]["side"] = 2
            elif change == "saved":
                self.f.payloads[0]["saved"]["history"] = []
            elif change == "storage":
                self.f.payloads[0]["raw"]["storageError"] = "save failed"
            elif change == "outcome":
                self.f.result["outcome"] = "win"
            else:
                self.f.result["canonical_trajectory"] = "0" * 64
            self.f.write()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.f.verify()
            self.f.result.update(verify_terminal_game(state(self.f.history, self.f.ai_side)))

    def test_identical_recovered_events_merge_but_conflicts_and_missing_seq_fail(self):
        self.f.add_payload(self.f.events)
        audit = self.f.verify()
        self.assertEqual(audit["duplicate_events_merged"], len(self.f.events))
        self.f.payloads[1]["audit"]["events"][0]["at"] += 1
        self.f.write(1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.f.verify()
        self.f.payloads[1]["audit"]["events"] = []
        self.f.write(1)
        self.f.payloads[0]["audit"]["events"].pop(2)
        self.f.write()
        with self.assertRaisesRegex(ValueError, "sequence contains a gap"):
            self.f.verify()

    def test_reload_new_page_can_resume_same_prefix_without_losing_actions(self):
        events = self.f.payloads[0]["audit"]["events"]
        cut = next(i for i, e in enumerate(events) if e["kind"] == "position"
                   and len(e["data"]["state"]["history"]) == 2)
        latter = copy.deepcopy(events[cut:])
        self.f.payloads[0]["audit"]["events"] = events[:cut+1]
        self.f.write()
        for index, event in enumerate(latter):
            event.update(seq=index+1, at=float(index+1))
        self.f.add_payload(latter, page="page-after-reload")
        self.assertEqual(self.f.verify()["pages"], 2)

    def test_configuration_empty_not_started_cannot_replace_committed_opening(self):
        original = copy.deepcopy(self.f.payloads[0])
        events = self.f.payloads[0]["audit"]["events"]
        setup = copy.deepcopy(events[0])
        setup["data"]["state"]["started"] = False
        setup["data"]["saved"] = None
        events.insert(0, setup)
        for index, event in enumerate(events):
            event.update(seq=index+1, at=float(index+1))
        self.f.write()
        self.assertEqual(self.f.verify()["ignored_setup_positions"], 1)
        self.f.payloads[0] = copy.deepcopy(original)
        self.f.mutate_event("position", lambda e: e["data"]["state"].__setitem__("human", 2))
        with self.assertRaisesRegex(ValueError, "prefixes"):
            self.f.verify()

    def test_visible_reset_after_moves_and_foreign_nonempty_position_are_rejected(self):
        original = copy.deepcopy(self.f.payloads[0])
        events = self.f.payloads[0]["audit"]["events"]
        target = next(e for e in events if e["kind"] == "position"
                      and len(e["data"]["state"]["history"]) == 3)
        target["data"] = copy.deepcopy(events[0]["data"])
        self.f.write()
        with self.assertRaisesRegex(ValueError, "rolled back"):
            self.f.verify()
        self.f.payloads[0] = copy.deepcopy(original)
        events = self.f.payloads[0]["audit"]["events"]
        target = next(e for e in events if e["kind"] == "position"
                      and len(e["data"]["state"]["history"]) == 1)
        target["data"]["state"]["human"] = 2
        self.f.write()
        with self.assertRaises(ValueError):
            self.f.verify()

    def test_no_evidence_or_no_terminal_capture_is_rejected(self):
        references = self.f.result["evidence"]
        self.f.result["evidence"] = []
        with self.assertRaises(ValueError):
            self.f.verify()
        self.f.result["evidence"] = references
        payload = self.f.payloads[0]
        payload["raw"] = raw([], self.f.ai_side)
        payload["state"] = state([], self.f.ai_side)
        payload["saved"] = copy.deepcopy(payload["raw"])
        self.f.write()
        with self.assertRaisesRegex(ValueError, "terminal position"):
            self.f.verify()


if __name__ == "__main__":
    unittest.main()
