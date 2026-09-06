"""Audit frozen-browser game artifacts without running a browser or any search.

This verifies recorded UI actions and legal game outcomes. Worker proof values
and representative PVs are retained as search observations, never accepted as
complete defender-branch certificates or as substitutes for a played result.
"""
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.browser.match_verify import (
    N, CELL_COUNT, _cells, _integer, _object, _replay, _required, _seconds,
    _side, _verified_normalized, file_hash, verify_browser_state,
    verify_terminal_game,
)
from web_match_runner import canonical_json

from tools.browser.search_budget import (
    LEGACY_AI_NODES, acceptance_node_limit, frozen_search_budget,
)

MAX_AI_NODES = LEGACY_AI_NODES  # Historical default; never relax existing runs.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _sha(value):
    value = _text(value, "sha256")
    if len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError("Invalid SHA256 digest")
    return value.lower()


def _path(base, value):
    value = _text(value, "evidence path")
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"Missing file or path outside the evidence root: {value}")
    return path


def _checked_file(base, value, expected):
    path = _path(base, value)
    if file_hash(path) != _sha(expected):
        raise ValueError(f"File hash mismatch: {value}")
    return path


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Invalid evidence JSON") from exc


def _indices(state):
    return [move["row"] * N + move["col"] for move in state["history"]]


def _flat(board):
    return [cell for row in board for cell in row]


def _same(left, right):
    # JSON comparison distinguishes bool from int, unlike Python list equality.
    return canonical_json(left) == canonical_json(right)


def _prefix(state, final):
    if state["ai_side"] != final["ai_side"] or state["history"] != final["history"][:state["plies"]]:
        raise ValueError("Evidence is not a prefix of this complete game")
    return state


def _saved(saved, raw):
    if isinstance(saved, str):
        saved = _json(saved)
    saved = _object(saved, "saved browser storage")
    saved_rows = _integer(saved.get("n", raw["n"]), "saved n", N, N)
    saved_cols = _integer(saved.get("cols", saved_rows), "saved cols", N, N)
    visible_cols = _integer(raw.get("cols", raw["n"]), "visible cols", N, N)
    if saved_cols != visible_cols:
        raise ValueError("Saved columns differ from the visible state")
    board = _cells(_required(saved, "board"), flattened=True)
    history = _required(saved, "history")
    if not isinstance(history, list):
        raise ValueError("Saved history must be a list")
    for point in history:
        _integer(point, "saved history point", 0, CELL_COUNT - 1)
    if board != _cells(raw["board"], flattened=True) or not _same(history, raw["history"]):
        raise ValueError("Saved board/history differ from the visible browser state")
    for field in ("n", "human", "seconds", "started"):
        if field in saved and not _same(saved[field], raw[field]):
            raise ValueError(f"Saved {field} differs from the visible state")


def _point(value):
    if isinstance(value, str):
        if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
            raise ValueError("Input point must be a canonical decimal cell index")
        value = int(value)
    return _integer(value, "input point", 0, CELL_COUNT - 1)


def _move_pair(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a row/column list")
    return [_integer(value[0], name + " row", 0, N - 1),
            _integer(value[1], name + " col", 0, N - 1)]


def _request(value, seconds):
    value = _object(value, "analysis request")
    if _required(value, "type") != "analyze":
        raise ValueError("Worker request is not an analysis")
    _integer(_required(value, "id"), "request id", 0, 2**53 - 1)
    rows = _integer(_required(value, "n"), "request n", N, N)
    _integer(value.get("cols", rows), "request cols", N, N)
    _side(_required(value, "side"), "request side")
    _cells(_required(value, "board"), flattened=True)
    actual = _seconds(_required(value, "seconds"), "request seconds")
    if not math.isclose(actual, seconds, rel_tol=0, abs_tol=1e-12):
        raise ValueError("AI request changed the frozen time budget")
    return value


def _ai_result(value, request, seconds, budget=None):
    value = _object(value, "analysis result")
    if _required(value, "type") != "result":
        raise ValueError("Worker response is not a result")
    if _integer(_required(value, "id"), "result id", 0, 2**53 - 1) != request["id"]:
        raise ValueError("Worker request/result identities differ")
    if "terminal" in value and value["terminal"] is not False:
        raise ValueError("A terminal analysis cannot account for a played AI move")
    if _required(value, "valueUsedInSearch") is not False:
        raise ValueError("Neural value was used in candidate search")
    if not math.isclose(_seconds(_required(value, "seconds"), "result seconds"), seconds,
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("AI result changed the frozen time budget")
    search = _object(_required(value, "search"), "AI search")
    _integer(_required(search, "move"), "AI selected move", 0, CELL_COUNT - 1)
    node_limit = acceptance_node_limit(budget)
    nodes = _integer(_required(search, "nodes"), "AI nodes", 0, node_limit)
    if budget is not None:
        _integer(_required(search, "budgetVersion"), "search budget version", budget["version"], budget["version"])
        _integer(_required(search, "nodeLimit"), "search node limit", node_limit, node_limit)
    elapsed = _number(_required(value, "elapsedMs"), "elapsedMs")
    inference = _number(_required(value, "inferenceMs"), "inferenceMs")
    overrun = _number(_required(value, "overrunMs"), "overrunMs")
    if inference > elapsed + 1e-6:
        raise ValueError("Inference duration exceeds total elapsed time")
    if not math.isclose(overrun, max(0.0, elapsed - seconds * 1000), rel_tol=1e-9, abs_tol=1e-5):
        raise ValueError("AI overrun accounting is inconsistent")
    return dict(nodes=nodes, elapsed_ms=elapsed, inference_ms=inference,
                search_ms=max(0.0, elapsed - inference), overrun_ms=overrun)


def verify_game_evidence(run, manifest, result):
    """Return a deterministic audit, or raise ValueError for incomplete evidence.

    File and screenshot references are relative to run (absolute paths inside run
    are also accepted). Repeated page_id/seq events after interruption are merged
    only when their complete contents agree. Unstarted empty setup observations
    cannot satisfy a played prefix; every started prefix needs matching DOM and
    persisted board/history. All stored files remain read-only.
    """
    run = Path(run).resolve()
    if not run.is_dir():
        raise ValueError("Run directory does not exist")
    manifest, result = _object(manifest, "manifest"), _object(result, "result")
    identity = {name: _text(_required(manifest, name), name) for name in ("run_id", "candidate_id")}
    game_id = _text(_required(result, "game_id"), "game_id")
    seconds = _seconds(_required(manifest, "seconds"), "manifest seconds")
    if seconds != 1.0:
        raise ValueError("This acceptance requires the frozen one-second budget")
    for name, expected in identity.items():
        if name in result and result[name] != expected:
            raise ValueError(f"Result {name} differs from the manifest")
    final = verify_terminal_game(result)
    for field in ("outcome", "termination", "canonical_trajectory"):
        if _required(result, field) != final[field]:
            raise ValueError(f"Claimed result {field} differs from legal replay")
    web_hashes = _object(_required(manifest, "web_hashes"), "web_hashes")
    if not web_hashes:
        raise ValueError("Frozen web hashes are missing")
    for name, checksum in web_hashes.items():
        _checked_file(run / "frozen" / "web", name, checksum)

    budget = frozen_search_budget(run / "frozen" / "web", web_hashes, manifest.get("search_budget"))
    references = _required(result, "evidence")
    if not isinstance(references, list) or not references:
        raise ValueError("A terminal game needs browser evidence files")
    pages, files_seen, duplicate_events, terminal_capture = {}, set(), 0, False
    evidence_files = []
    for reference in references:
        reference = _object(reference, "evidence reference")
        if "game_id" in reference and reference["game_id"] != game_id:
            raise ValueError("Evidence reference belongs to another game")
        path = _checked_file(run, _required(reference, "file"), _required(reference, "sha256"))
        if path in files_seen:
            raise ValueError("The same evidence file was listed twice")
        files_seen.add(path)
        png = _checked_file(run, _required(reference, "screenshot"),
                            _required(reference, "screenshot_sha256"))
        if path.suffixes[-2:] != [".json", ".gz"] or png.suffix.lower() != ".png":
            raise ValueError("Evidence must be gzip JSON with a PNG screenshot")
        with png.open("rb") as stream:
            if stream.read(8) != PNG_SIGNATURE:
                raise ValueError("Screenshot is not a PNG")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = _object(_json(stream.read()), "evidence payload")
        except (OSError, EOFError, UnicodeError) as exc:
            raise ValueError("Invalid compressed evidence file") from exc
        for name, expected in dict(identity, game_id=game_id).items():
            if _required(payload, name) != expected:
                raise ValueError(f"Evidence {name} differs from this game")
        _text(_required(payload, "stage"), "evidence stage")
        if _path(run, _required(payload, "screenshot")) != png:
            raise ValueError("Payload and reference name different screenshots")
        raw = _object(_required(payload, "raw"), "raw browser snapshot")
        state = _prefix(verify_browser_state(raw, final["ai_side"], seconds), final)
        if _verified_normalized(_required(payload, "state")) != state:
            raise ValueError("Evidence normalized state differs from its raw snapshot")
        if "saved" in payload:
            _saved(payload["saved"], raw)
        terminal_capture = terminal_capture or state["terminal"]
        filename = path.relative_to(run).as_posix()
        evidence_files.append(filename)
        audit = _object(_required(payload, "audit"), "audit")
        page = _text(_required(audit, "page_id"), "page_id")
        entries = _required(audit, "events")
        if not isinstance(entries, list):
            raise ValueError("Audit events must be a list")
        sequence = pages.setdefault(page, {})
        previous = 0
        for event in entries:
            event = _object(event, "audit event")
            seq = _integer(_required(event, "seq"), "event seq", 1, 2**53 - 1)
            _number(_required(event, "at"), "event time")
            _text(_required(event, "kind"), "event kind")
            _object(_required(event, "data"), "event data")
            if seq <= previous:
                raise ValueError("Events within a capture are not strictly ordered")
            previous = seq
            if seq in sequence:
                if not _same(event, sequence[seq][0]):
                    raise ValueError("Conflicting duplicate browser event")
                duplicate_events += 1
            else:
                sequence[seq] = (event, filename)
    if not terminal_capture:
        raise ValueError("No evidence capture contains the actual terminal position")

    events = []
    for page, sequence in pages.items():
        last_seq, last_at = None, -1.0
        for seq, (event, filename) in sorted(sequence.items()):
            if last_seq is not None and seq != last_seq + 1:
                raise ValueError("Browser audit sequence contains a gap")
            if event["at"] < last_at:
                raise ValueError("Browser event time moved backwards")
            last_seq, last_at = seq, event["at"]
            events.append(dict(event, page_id=page, evidence_file=filename, order=len(events)))
    positions, requests, inputs, analyses = {}, {}, [], []
    ignored_setup, last_ply = 0, 0
    for event in events:
        kind, data = event["kind"], event["data"]
        if kind == "position":
            raw = _object(_required(data, "state"), "position snapshot")
            # The initial page can have the other color selected. No played
            # position is allowed to change actors or reset its history.
            human = _side(_required(raw, "human"), "position human")
            normalized = verify_browser_state(raw, 3 - human, seconds)
            if not normalized["plies"] and (human != 3 - final["ai_side"] or not raw["started"]):
                if last_ply:
                    raise ValueError("A played game was reset during its evidence stream")
                ignored_setup += 1
                if human == 3 - final["ai_side"] and data.get("saved") is not None:
                    _saved(data["saved"], raw)
                continue
            normalized = _prefix(normalized, final)
            ply = normalized["plies"]
            if ply < last_ply:
                raise ValueError("Visible history rolled back in the browser")
            last_ply = ply
            cells = _required(data, "visible_cells")
            if not isinstance(cells, list) or len(cells) != CELL_COUNT:
                raise ValueError("DOM evidence must contain all 256 cells")
            visible, seen = [None] * CELL_COUNT, set()
            for cell in cells:
                cell = _object(cell, "visible cell")
                point = _integer(_required(cell, "point"), "visible point", 0, CELL_COUNT - 1)
                if point in seen:
                    raise ValueError("Duplicate DOM cell")
                seen.add(point)
                visible[point] = _integer(_required(cell, "cell"), "visible cell", 0, 2)
            if visible != raw["board"]:
                raise ValueError("DOM stones differ from the browser snapshot")
            _saved(_required(data, "saved"), raw)
            positions.setdefault(ply, []).append(event)
        elif kind == "analysis_requested":
            request = _request(data, seconds)
            key = (event["page_id"], request["id"])
            previous_requests = requests.setdefault(key, [])
            if previous_requests and not _same(previous_requests[0]["data"], request):
                raise ValueError("A worker request id was reused for a different position")
            previous_requests.append(event)
        elif kind == "analysis_result":
            analyses.append(event)
        elif kind == "input":
            inputs.append(event)
        # Browser errors remain in the artifact. They do not establish any move.
    if set(positions) != set(range(final["plies"] + 1)):
        missing = sorted(set(range(final["plies"] + 1)) - set(positions))
        raise ValueError(f"Missing visible and persisted position prefixes: {missing}")

    opponent = _required(result, "opponent_moves")
    if not isinstance(opponent, list):
        raise ValueError("opponent_moves must be a list")
    choices = {}
    for record in opponent:
        record = _object(record, "opponent record")
        if "game_id" in record and record["game_id"] != game_id:
            raise ValueError("Opponent record belongs to another game")
        before = _prefix(_verified_normalized(_required(record, "before")), final)
        ply = before["plies"]
        if before["terminal"] or ply >= final["plies"] or final["history"][ply]["by"] != "human":
            raise ValueError("Opponent search record is not before a real human turn")
        if ply in choices:
            raise ValueError("Duplicate opponent move record")
        choice = _object(_required(record, "choice"), "opponent choice")
        move = _move_pair(_required(choice, "move"), "opponent move")
        actual = final["history"][ply]
        if move != [actual["row"], actual["col"]]:
            raise ValueError("Opponent choice differs from the actual human action")
        search = _object(_required(choice, "search"), "opponent search")
        if _move_pair(_required(search, "move"), "opponent search move") != move:
            raise ValueError("Opponent action differs from its recorded search")
        _integer(_required(search, "nodes"), "opponent nodes", 0, 2**53 - 1)
        _text(_required(choice, "reason"), "opponent reason")
        if _required(choice, "challenger") != "full_board_native_vcf_challenger_v1":
            raise ValueError("Unknown challenger implementation")
        _sha(_required(choice, "root_order_seed"))
        if _required(choice, "neural_value_used") is not False:
            raise ValueError("Challenger unexpectedly used a neural value")
        options = _object(_required(choice, "options"), "opponent options")
        if "opponent_options" in manifest and not _same(options, manifest["opponent_options"]):
            raise ValueError("Challenger changed its frozen search options")
        choices[ply] = choice
    human_plies = {ply for ply, move in enumerate(final["history"]) if move["by"] == "human"}
    if set(choices) != human_plies:
        raise ValueError("Every actual human action needs one matching challenger record")

    move_audits, timings = [], []
    for ply, move in enumerate(final["history"]):
        before = _replay(_indices(final)[:ply], final["ai_side"])
        point = move["row"] * N + move["col"]
        start, end = positions[ply][0]["order"], positions[ply + 1][0]["order"]
        if end <= start:
            raise ValueError("Position evidence has an invalid action order")
        matched, accounting = None, None
        if move["by"] == "human":
            for event in inputs:
                data = event["data"]
                if (start < event["order"] < end and data.get("type") == "click"
                        and data.get("trusted") is True and _same(data.get("history"), _indices(before))
                        and data.get("point") is not None and _point(data["point"]) == point):
                    matched = event
                    break
        else:
            for event in analyses:
                if not start < event["order"] < end:
                    continue
                data = event["data"]
                request = _request(_required(data, "request"), seconds)
                if request["side"] != move["side"] or request["board"] != _flat(before["board"]):
                    continue
                response = _object(_required(data, "result"), "worker result")
                search = _object(_required(response, "search"), "worker search")
                if _integer(_required(search, "move"), "selected move", 0, CELL_COUNT - 1) != point:
                    continue
                requested = requests.get((event["page_id"], request["id"]), [])
                if not any(item["order"] < event["order"] and _same(item["data"], request)
                           for item in requested):
                    raise ValueError("AI result lacks its matching earlier worker request")
                accounting = _ai_result(response, request, seconds, budget)
                matched = event
                break
        if matched is None:
            raise ValueError(f"Move {ply + 1} lacks its trusted click or matching AI analysis")
        item = dict(ply=ply + 1, side=move["side"], by=move["by"],
                    move=[move["row"], move["col"]], page_id=matched["page_id"],
                    seq=matched["seq"], evidence_file=matched["evidence_file"])
        if accounting is not None:
            item.update(accounting)
            timings.append(accounting)
        move_audits.append(item)

    return dict(
        verified=True, run_id=identity["run_id"], candidate_id=identity["candidate_id"],
        game_id=game_id, ai_side=final["ai_side"], winner=final["winner"],
        outcome=final["outcome"], plies=final["plies"],
        canonical_trajectory=final["canonical_trajectory"],
        terminal_source="independent_legal_history_replay",
        evidence_files=evidence_files, screenshot_count=len(references),
        frozen_web_files_verified=len(web_hashes), pages=len(pages),
        unique_events=len(events), duplicate_events_merged=duplicate_events,
        ignored_setup_positions=ignored_setup, verified_prefixes=len(positions),
        human_moves=len(human_plies), ai_moves=len(timings), moves=move_audits,
        ai_budget=dict(seconds=seconds, max_nodes=acceptance_node_limit(budget),
                       total_nodes=sum(item["nodes"] for item in timings),
                       total_elapsed_ms=math.fsum(item["elapsed_ms"] for item in timings),
                       total_inference_ms=math.fsum(item["inference_ms"] for item in timings),
                       total_search_ms=math.fsum(item["search_ms"] for item in timings),
                       total_overrun_ms=math.fsum(item["overrun_ms"] for item in timings),
                       moves_over_budget=sum(item["overrun_ms"] > 0 for item in timings)),
        complete_branch_certificates_verified=False,
        proof_evidence_limit="Search proof flags and representative PVs do not provide every defender branch; only the actually played terminal outcome is verified.",
    )
