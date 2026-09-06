"""Isolated, resumable real-browser Gomoku acceptance. Mutations use trusted CDP clicks only."""
import argparse, base64, gzip, hashlib, importlib, importlib.metadata, json, os
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import queue, shutil, socket, subprocess, sys, threading, time, traceback, urllib.request, uuid
import numpy as np

ROOT = Path(__file__).resolve().parent
PROTECTED_PORTS = {8765, 8766}
TARGETS = {"ai_first": 1000, "ai_second": 1000}
FORMAT = "gomoku_real_browser_acceptance_v1"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            result.update(data)
    return result.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers/scanners can briefly hold a handle without delete sharing.
        # Retry only the rename; the same complete temporary file is already fsynced.
        deadline = time.monotonic() + 0.5
        delays = (0.025, 0.05, 0.075, 0.1, 0.15)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                if (getattr(exc, "winerror", None) not in (5, 32, 33)
                        or attempt == len(delays)
                        or time.monotonic() + delays[attempt] >= deadline):
                    raise
                time.sleep(delays[attempt])
                if time.monotonic() >= deadline:
                    raise
    finally:
        temporary.unlink(missing_ok=True)


class Journal:
    def __init__(self, path):
        self.path, self.events = Path(path), []
        previous = "0"*64
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                checksum = item.pop("sha256")
                if item["seq"] != len(self.events)+1 or item["previous"] != previous or digest(item) != checksum:
                    raise ValueError("Journal integrity failure; existing results cannot be replaced")
                item["sha256"] = checksum
                self.events.append(item)
                previous = checksum

    def append(self, kind, data):
        item = {"seq": len(self.events)+1, "previous": self.events[-1]["sha256"] if self.events else "0"*64,
                "utc": utc(), "kind": kind, "data": data}
        item["sha256"] = digest(item)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(item)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(item)
        return item


def winner(board):
    board = np.asarray(board)
    for side in (1, 2):
        for row, col in np.argwhere(board == side):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = int(row)+4*dr, int(col)+4*dc
                if 0 <= er < board.shape[0] and 0 <= ec < board.shape[1]:
                    if all(board[int(row)+k*dr, int(col)+k*dc] == side for k in range(5)):
                        return side
    return 0


def verify_state(state):
    board = np.asarray(state["board"])
    if board.shape != (16, 16) or board.dtype.kind not in "iu" or not np.isin(board, [0, 1, 2]).all():
        raise ValueError("Acceptance requires integer 16x16 boards without forbidden cells")
    ai, human = state["ai_side"], state["human_side"]
    if isinstance(ai, bool) or ai not in (1, 2) or human != 3-ai:
        raise ValueError("Invalid player identity")
    rebuilt = np.zeros((16, 16), dtype=np.uint8)
    for index, move in enumerate(state["history"]):
        side, row, col = move["side"], move["row"], move["col"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (side, row, col)):
            raise ValueError("Noninteger history")
        if side != 1+index%2 or not 0 <= row < 16 or not 0 <= col < 16:
            raise ValueError("Wrong turn/coordinate")
        if winner(rebuilt) or rebuilt[row, col] or move["by"] != ("ai" if side == ai else "human"):
            raise ValueError("Illegal move or continuation after terminal")
        rebuilt[row, col] = side
    if not np.array_equal(rebuilt, board) or state["move_count"] != len(state["history"]):
        raise ValueError("Board differs from complete history")
    won = winner(board)
    terminal = bool(won or not np.any(board == 0))
    if state["winner"] != won or bool(state["finished"]) != terminal:
        raise ValueError("Terminal status differs from actual five/full-board rules")
    if not terminal and (state["turn"] != human or 1+len(state["history"])%2 != human):
        raise ValueError("Response is not a complete human turn")
    return board.astype(np.uint8), won, terminal


def trajectory_key(history, ai_side):
    forms = []
    for rotation in range(4):
        for reflection in (False, True):
            for swap in (False, True):
                moves = []
                for move in history:
                    row, col = move["row"], move["col"]
                    for _ in range(rotation):
                        row, col = col, 15-row
                    if reflection:
                        col = 15-col
                    moves.append([3-move["side"] if swap else move["side"], row, col])
                forms.append(canonical_json({"ai_side": 3-ai_side if swap else ai_side, "moves": moves}))
    return hashlib.sha256(min(forms).encode("utf-8")).hexdigest()


def summarize(manifest, events, error=None, *, running=False):
    finished = [e["data"] for e in events if e["kind"] == "game_completed"]
    seen, unique, duplicates = set(), [], 0
    for game in finished:
        if game["canonical_trajectory"] in seen:
            duplicates += 1
        else:
            seen.add(game["canonical_trajectory"])
            unique.append(game)
    black_all, white_all = ([g for g in unique if g["ai_side"] == side] for side in (1, 2))
    black, white = black_all[:1000], white_all[:1000]
    losses = sum(g["ai_side"] == 1 and g["outcome"] == "loss" for g in finished)
    rate = sum(g["outcome"] in ("win", "draw") for g in white)/len(white) if white else None
    white_failed = bool(manifest["formal"] and len(white)==1000 and rate<=.5)
    done = bool(manifest["formal"] and len(black)==1000 and len(white)==1000 and not losses and rate>.5)
    status = "candidate_failed" if losses or white_failed else "run_error" if error else "goal_met" if done else (
        "running" if running else "paused")
    return {"format": FORMAT, "run_id": manifest["run_id"], "candidate_id": manifest["candidate_id"],
            "formal": manifest["formal"], "counts_toward_formal_goal": manifest["formal"], "targets": TARGETS,
            "status": status, "goal_met": done, "completed_attempts": len(finished),
            "duplicate_complete_games": duplicates, "ai_first_unique_games": len(black),
            "ai_second_unique_games": len(white), "black_losses": losses, "white_nonloss_rate": rate,
            "outcomes": {str(side): {result: sum(g["ai_side"]==side and g["outcome"]==result for g in black+white)
                        for result in ("win", "draw", "loss")} for side in (1, 2)},
            "white_fixed_sample_failed": white_failed, "scoring_rule": "first_1000_unique_per_ai_color",
            "all_unique_games": {"ai_first": len(black_all), "ai_second": len(white_all)},
            "error": error, "updated_utc": utc(),
            "interpretation": "Actual terminal games only. Score the first 1000 unique games per AI color. Duplicates add no coverage. Every black-first loss remains a failure; a failed fixed white sample cannot be rescued by extra games."}


def next_ai_side(manifest, events, attempt_index):
    status = summarize(manifest, events)
    if status["status"] == "candidate_failed" or status["goal_met"]:
        return None
    if manifest["formal"]:
        if status["ai_first_unique_games"] == 1000:
            return 2
        if status["ai_second_unique_games"] == 1000:
            return 1
    return 1 if attempt_index % 2 == 0 else 2


class JsonProcess:
    def __init__(self, args, cwd, error_path):
        self.error_file = Path(error_path).open("ab")
        self.process = subprocess.Popen(args, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.error_file, text=True, encoding="utf-8", bufsize=1,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        self.inbox, self.counter = queue.Queue(), 0
        def read():
            try:
                for line in self.process.stdout:
                    self.inbox.put(json.loads(line))
            except Exception as exc:
                self.inbox.put({"fatal": str(exc)})
            self.inbox.put({"fatal": "Helper ended"})
        threading.Thread(target=read, daemon=True).start()
        try:
            self.ready = self.inbox.get(timeout=60)
            if not self.ready.get("ready"):
                raise RuntimeError(f"Helper initialization failed: {self.ready}")
        except BaseException:
            self.close()
            raise

    def request(self, payload, timeout=120):
        self.counter += 1
        self.process.stdin.write(canonical_json({**payload, "id": self.counter})+"\n")
        self.process.stdin.flush()
        result = self.inbox.get(timeout=timeout)
        if result.get("fatal") or result.get("error") or result.get("id") != self.counter:
            raise RuntimeError(f"Helper failed: {result}")
        return result["result"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.error_file.close()


def worker_main(callback, options, fingerprint_only=False):
    module_name, name = callback.split(":", 1)
    with redirect_stdout(sys.stderr):
        module = importlib.import_module(module_name)
        choose = getattr(module, name)
        fingerprint = module.configuration_fingerprint(options) if hasattr(module, "configuration_fingerprint") else {
            "module": callback, "options": options, "source_sha256": file_hash(module.__file__)}
    print(canonical_json({"ready": True, "fingerprint": fingerprint}), flush=True)
    if fingerprint_only:
        return 0
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with redirect_stdout(sys.stderr):
                result = choose(np.asarray(request["board"], dtype=np.uint8), request["side"], context=request["context"])
            if not isinstance(result, dict):
                result = {"move": list(result)}
            result = json.loads(json.dumps(result, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()))
            print(canonical_json({"id": request["id"], "result": result}), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(canonical_json({"id": request["id"], "error": str(exc)}), flush=True)
    return 0


def locate_browser():
    for path in (Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
                 Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")):
        if path.exists():
            return str(path)
    raise RuntimeError("Installed Chromium browser required; pass --browser")


def strip_paths(value):
    if isinstance(value, dict):
        return {k: strip_paths(v) for k, v in value.items() if not k.endswith("path") and k!="source"}
    if isinstance(value, list):
        return [strip_paths(v) for v in value]
    return value


def freeze_run(args):
    run = Path(args.run_dir).resolve()
    if run.exists():
        raise ValueError("Run directory exists; use --resume. Existing candidates/results cannot be overwritten.")
    sources = run/"frozen"/"source"
    sources.mkdir(parents=True)
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".mjs", ".c", ".h")]
    files.append(ROOT/"web"/"unet_game.html")
    native = ROOT/"native_board.c"
    if native.exists():
        folder = ROOT/"exports"/"native_search"/file_hash(native)[:16]
        if args.engine in ("native", "guarded") and not (folder/"native_board.dll").exists():
            raise ValueError("Reviewed native DLL must be built before freezing")
        if folder.exists():
            files += [p for p in folder.iterdir() if p.suffix in (".dll", ".json")]
    source_hashes = {}
    for original in files:
        relative = original.relative_to(ROOT)
        target = sources/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        source_hashes[str(relative).replace("\\", "/")] = file_hash(target)
    models = run/"frozen"/"models"
    models.mkdir()
    paths = {"opponent.pt": Path(args.models)/"opponent.pt", "play.pt": Path(args.models)/"play.pt"}
    if args.strategy:
        paths["global.pt"] = Path(args.strategy)
    model_hashes = {}
    for name, original in paths.items():
        shutil.copyfile(original.resolve(), models/name)
        model_hashes[name] = file_hash(models/name)
    options = json.loads(Path(args.opponent_options_file).read_text(encoding="utf-8")) if args.opponent_options_file else json.loads(args.opponent_options)
    module = args.opponent.split(":", 1)[0]
    if not (sources/(module.replace(".", "/")+".py")).exists():
        raise ValueError("Opponent must be a local module included in the frozen snapshot")
    process = subprocess.run([sys.executable, "-B", str(sources/"web_match_runner.py"),
        "--worker-callback", args.opponent, "--worker-options-json", canonical_json(options), "--fingerprint-only"],
        cwd=sources, capture_output=True, text=True, encoding="utf-8", timeout=60,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    if process.returncode:
        raise RuntimeError(process.stderr)
    opponent = json.loads(process.stdout)["fingerprint"]
    resolved = opponent.get("options", options)
    for field, limit in (("time_limit", args.search_seconds), ("max_nodes", args.search_nodes),
                         ("depth", args.search_depth), ("candidate_width", args.search_width)):
        if field not in resolved or resolved[field] < limit:
            raise ValueError(f"Challenger {field} must be >= candidate setting {limit}")
    node = str(Path(args.node or shutil.which("node") or "").resolve())
    browser = str(Path(args.browser or locate_browser()).resolve())
    if not Path(node).is_file() or not Path(browser).is_file():
        raise ValueError("Node.js and Chromium paths must exist")
    version = subprocess.check_output([node, "-p", "JSON.stringify({version:process.version,websocket:typeof WebSocket})"], text=True).strip()
    if json.loads(version)["websocket"]!="function":
        raise ValueError("Node with built-in WebSocket is required")
    config = {k: getattr(args, k) for k in ("engine", "search_seconds", "search_nodes", "search_depth", "search_width",
              "forcing_seconds", "forcing_nodes", "forcing_depth", "value_weight", "threads",
               "guard_native_fraction", "guard_probe_nodes", "guard_probe_seconds",
              "guard_threat_seconds", "guard_threat_nodes", "guard_threat_width",
              "guard_threat_quiet_plies", "guard_threat_total_plies", "guard_attack_seconds", "guard_attack_nodes") if getattr(args, k) is not None}
    config["max_size"] = 16
    candidate = {"weights": model_hashes, "sources": source_hashes, "server": config,
                 "opponent": strip_paths(opponent), "protocol": FORMAT}
    manifest = {"format": FORMAT, "run_id": uuid.uuid4().hex, "candidate_id": digest(candidate),
        "formal": args.formal, "created_utc": utc(), "targets": TARGETS, "seed": args.seed,
        "source_root": str(sources), "model_root": str(models), "source_hashes": source_hashes,
        "model_hashes": model_hashes, "server": config, "opponent_callback": args.opponent,
        "opponent_options": resolved, "opponent_fingerprint": opponent,
        "browser": browser, "browser_sha256": file_hash(browser), "node": node, "node_sha256": file_hash(node),
        "node_version": json.loads(version), "python": sys.executable, "python_version": sys.version,
        "numpy_version": np.__version__, "torch_version": importlib.metadata.version("torch"),
        "headed": args.headed, "registry": str((ROOT/"web_acceptance"/"candidate_registry.jsonl").resolve())}
    with (run/"manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(manifest))
    (run/"runtime").mkdir()
    return manifest


def verify_frozen(manifest):
    for relative, expected in manifest["source_hashes"].items():
        if file_hash(Path(manifest["source_root"])/relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    for name, expected in manifest["model_hashes"].items():
        if file_hash(Path(manifest["model_root"])/name) != expected:
            raise ValueError(f"Frozen weights changed: {name}")
    for name in ("browser", "node"):
        if file_hash(manifest[name]) != manifest[name+"_sha256"]:
            raise ValueError(f"Frozen {name} binary changed")
    if (sys.version != manifest["python_version"] or np.__version__ != manifest["numpy_version"]
            or importlib.metadata.version("torch") != manifest["torch_version"]):
        raise ValueError("Frozen Python/numpy/torch environment changed")



def verify_persisted_identity(manifest, identity):
    if identity.get("run_id") != manifest["run_id"]:
        raise ValueError("Persisted run identity changed")
    for field, value in manifest["server"].items():
        if identity["configuration"].get(field) != value:
            raise ValueError(f"Persisted server setting differs: {field}")
    for filename, checksum in manifest["model_hashes"].items():
        if identity["weights"].get(Path(filename).stem) != checksum:
            raise ValueError("Persisted model hash differs")
    expected_sources = {name: checksum for name, checksum in manifest["source_hashes"].items()
                        if "/" not in name and Path(name).suffix in (".py", ".c", ".dll", ".mjs")}
    if identity["sources"] != expected_sources:
        raise ValueError("Persisted server source identity differs")
    if manifest["server"]["engine"] in ("native", "guarded"):
        binary = next(checksum for name, checksum in manifest["source_hashes"].items()
                      if name.endswith("/native_board.dll"))
        if identity["native"]["binary_sha256"] != binary:
            raise ValueError("Persisted native DLL hash differs")


def free_port():
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in PROTECTED_PORTS:
            return port


def listening(port):
    if port in PROTECTED_PORTS:
        raise ValueError("Active user-game ports are prohibited")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.2):
            return True
    except OSError:
        return False



class Runtime:
    def __init__(self, run, manifest, journal):
        self.run, self.manifest, self.journal = Path(run), manifest, journal
        self.server = self.browser = self.bridge = self.opponent = None
        self.logs, self.receipts = [], []
        record_file = self.run/"runtime"/"server.json"
        old = json.loads(record_file.read_text()) if record_file.exists() else None
        if old and old["run_id"] == manifest["run_id"] and listening(old["port"]):
            port = old["port"]
        else:
            port, config, models = free_port(), manifest["server"], Path(manifest["model_root"])
            args = [manifest["python"], "-B", str(Path(manifest["source_root"])/"play_unet.py"),
                    "--models", str(models), "--port", str(port),
                    "--state-file", str(self.run/"runtime"/"game_state.json"), "--run-id", manifest["run_id"]]
            if "global.pt" in manifest["model_hashes"]:
                args += ["--strategy", str(models/"global.pt")]
            for name, value in config.items():
                args += ["--"+name.replace("_", "-"), str(value)]
            log = (self.run/"runtime"/"server.log").open("ab")
            self.logs.append(log)
            self.server = subprocess.Popen(args, cwd=manifest["source_root"], stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            atomic_json(record_file, {"run_id": manifest["run_id"], "port": port, "pid": self.server.pid, "argv": args})
            journal.append("server_started", {"pid": self.server.pid, "port": port, "argv": args})
            deadline = time.monotonic()+60
            while not listening(port):
                if self.server.poll() is not None:
                    raise RuntimeError("Isolated persisted server failed; see runtime/server.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("Isolated server did not start")
                time.sleep(.1)
        self.origin = f"http://127.0.0.1:{port}"
        profile = self.run/"runtime"/("browser_"+uuid.uuid4().hex)
        profile.mkdir()
        args = [manifest["browser"], "--no-first-run", "--no-default-browser-check",
                "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
                "--user-data-dir="+str(profile), "--window-size=1400,1100", "about:blank"]
        if not manifest["headed"]:
            args.insert(1, "--headless=new")
        log = (profile/"browser.log").open("ab")
        self.logs.append(log)
        self.browser = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        active = profile/"DevToolsActivePort"
        deadline = time.monotonic()+30
        while not active.exists():
            if self.browser.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError("Dedicated Edge did not expose its debugging port")
            time.sleep(.1)
        debug_port = int(active.read_text().splitlines()[0])
        pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json/list", timeout=5))
        page = next(p for p in pages if p["type"]=="page")
        self.bridge = JsonProcess([manifest["node"], str(Path(manifest["source_root"])/"web_match_cdp.mjs"),
                                   page["webSocketDebuggerUrl"], self.origin], manifest["source_root"], profile/"cdp.log")
        self.cdp("Page.addScriptToEvaluateOnNewDocument", {"source": """
          window.__acceptanceInput=[];
          for(const name of ['click','keydown'])document.addEventListener(name,e=>{
            window.__acceptanceInput.push({type:e.type,trusted:e.isTrusted,target:e.target.id||e.target.tagName,
              x:e.clientX||0,y:e.clientY||0,key:e.key||'',at:performance.now()});
            if(window.__acceptanceInput.length>20)window.__acceptanceInput.shift();
          },true);
        """})
        self.cdp("Page.navigate", {"url": self.origin+"/"})
        self.wait_state()
        self.opponent = JsonProcess([manifest["python"], "-B", str(Path(manifest["source_root"])/"web_match_runner.py"),
            "--worker-callback", manifest["opponent_callback"], "--worker-options-json", canonical_json(manifest["opponent_options"])],
            manifest["source_root"], self.run/"runtime"/"opponent.log")
        if strip_paths(self.opponent.ready["fingerprint"]) != strip_paths(manifest["opponent_fingerprint"]):
            raise ValueError("Frozen challenger configuration changed")

    def cdp(self, method, params=None):
        return self.bridge.request({"method": method, "params": params or {}})

    def evaluate(self, expression):
        result = self.cdp("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise RuntimeError(str(result["exceptionDetails"]))
        return result["result"].get("value")

    def snapshot(self):
        return self.evaluate("""(() => {
          if(typeof state==='undefined'||!state)return null;
          const canvas=document.getElementById('board'),r=canvas.getBoundingClientRect();
          return {state:JSON.parse(JSON.stringify(state)),busy:typeof busy!=='undefined'&&busy,
           input_events:window.__acceptanceInput||[],drafting:typeof drafting!=='undefined'&&drafting,url:location.href,
           dom:{title:document.title,turn:document.getElementById('turnTitle').textContent,
           detail:document.getElementById('boardDetail').textContent,history:document.getElementById('history').textContent,
           model_source:document.getElementById('source').textContent,error:document.getElementById('error').textContent,
           selected_first:document.getElementById('first').value},
           canvas:{x:r.x,y:r.y,pageX:r.x+scrollX,pageY:r.y+scrollY,width:r.width,height:r.height,logicalWidth,logicalHeight,cell,padding}};
        })()""")

    def wait_state(self, timeout=120):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:
                snap = self.snapshot()
                self.receipts += self.cdp("__receipts")["receipts"]
                if snap and not snap["busy"]:
                    state = snap["state"]
                    verify_state(state)
                    session = state.get("session", {})
                    if not session.get("persisted") or session.get("run_id") != self.manifest["run_id"]:
                        raise ValueError("Browser is not connected to the frozen acceptance game")
                    persisted = json.loads((self.run/"runtime"/"game_state.json").read_text(encoding="utf-8"))
                    verify_persisted_identity(self.manifest, persisted["identity"])
                    if session["configuration_sha256"] != digest(persisted["identity"]):
                        raise ValueError("Browser/persistence identity mismatch")
                    if str(Path(state["model_source"]).resolve()) != str(Path(self.manifest["model_root"]).resolve()):
                        raise ValueError("Server did not load this run's frozen models")
                    if not all(state["models"][role]["trained"] for role in ("opponent", "play")):
                        raise ValueError("Untrained local models are not accepted")
                    if "global.pt" in self.manifest["model_hashes"] and not state["models"]["global"]["trained"]:
                        raise ValueError("Untrained global model is not accepted")
                    return snap
            except RuntimeError:
                pass
            time.sleep(.05)
        raise TimeoutError("Page did not return a verified complete state")

    def click_xy(self, x, y):
        for kind in ("mousePressed", "mouseReleased"):
            self.cdp("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, "button": "left", "clickCount": 1})

    def click_element(self, element_id):
        point = self.evaluate(f"(() => {{const e=document.getElementById({json.dumps(element_id)});e.scrollIntoView({{block:'center'}});const r=e.getBoundingClientRect();return {{x:r.x+r.width/2,y:r.y+r.height/2}};}})()")
        self.click_xy(point["x"], point["y"])

    def key(self, key, code):
        for kind in ("keyDown", "keyUp"):
            self.cdp("Input.dispatchKeyEvent", {"type": kind, "key": key, "code": key,
                                              "windowsVirtualKeyCode": code, "nativeVirtualKeyCode": code})

    def choose_color(self, ai_side):
        self.click_element("first")
        self.key("Home" if ai_side==1 else "End", 36 if ai_side==1 else 35)
        self.key("Enter", 13)
        if self.evaluate("document.getElementById('first').value") != ("ai" if ai_side==1 else "human"):
            raise RuntimeError("Real keyboard input did not select the requested player color")

    def click_cell(self, row, col):
        self.evaluate("document.getElementById('board').scrollIntoView({block:'center',inline:'center'})")
        c = self.snapshot()["canvas"]
        self.click_xy(c["x"]+(c["padding"]+(col+.5)*c["cell"])*c["width"]/c["logicalWidth"],
                      c["y"]+(c["padding"]+(row+.5)*c["cell"])*c["height"]/c["logicalHeight"])

    def find_receipt(self, snapshot):
        self.receipts += self.cdp("__receipts")["receipts"]
        return next((r for r in reversed(self.receipts) if r.get("status")==200 and r.get("body")==snapshot["state"]), None)

    def capture(self, path, snapshot):
        c = snapshot["canvas"]
        result = self.cdp("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True,
            "clip": {"x": max(c["pageX"], 0), "y": max(c["pageY"], 0), "width": c["width"], "height": c["height"], "scale": 1}})
        with Path(path).open("xb") as stream:
            stream.write(base64.b64decode(result["data"]))
        return file_hash(path)

    def close(self, keep_server=False):
        if self.bridge:
            try:
                self.cdp("Browser.close")
            except Exception:
                pass
            self.bridge.close()
        if self.opponent:
            self.opponent.close()
        if self.browser and self.browser.poll() is None:
            self.browser.terminate()
        if self.server and self.server.poll() is None and not keep_server:
            self.server.terminate()
            try:
                self.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server.kill()
        for stream in self.logs:
            stream.close()


def light_state(state):
    return {key: state[key] for key in ("board", "history", "revision", "ai_side", "human_side",
                                        "winner", "finished", "move_count", "turn", "session")}


def verify_transition(before, after, kind, move=None, ai_side=None):
    verify_state(after)
    if after["revision"] != before["revision"]+1:
        raise ValueError("A browser action must commit exactly one revision")
    if kind=="new":
        expected = [{"side": 1, "row": 8, "col": 8, "by": "ai"}] if ai_side==1 else []
        if after["ai_side"] != ai_side or after["history"] != expected or after["finished"]:
            raise ValueError("New game did not start from empty with the configured first player")
    else:
        prefix = before["history"]
        delta = after["history"][len(prefix):]
        if after["ai_side"] != before["ai_side"] or after["history"][:len(prefix)] != prefix or not delta:
            raise ValueError("Browser action replaced or skipped history")
        if (delta[0]["row"], delta[0]["col"]) != tuple(move) or delta[0]["by"] != "human":
            raise ValueError("Canvas click and committed human move differ")
        if len(delta) not in (1, 2) or len(delta)==1 and not after["finished"]:
            raise ValueError("Unexpected atomic-turn move count")


def save_evidence(run, journal, game_id, snapshot, receipt, intent, runtime, recovered=False):
    folder = Path(run)/"games"/game_id
    folder.mkdir(parents=True, exist_ok=True)
    stamp = f"{len(journal.events)+1:07d}_{uuid.uuid4().hex[:12]}"
    screenshot = folder/(stamp+".png")
    runtime.evaluate("document.getElementById('board').scrollIntoView({block:'center',inline:'center'})")
    snapshot = runtime.wait_state()
    if receipt["body"] != snapshot["state"]:
        raise ValueError("Page state and observed browser network receipt differ")
    if not recovered:
        clicks = [e for e in snapshot["input_events"] if e["type"]=="click"]
        if not clicks or not clicks[-1]["trusted"] or clicks[-1]["target"] != (
                "new" if intent["kind"]=="new" else "board"):
            raise ValueError("No trusted browser click for the confirmed action")
        if receipt.get("method") != "POST":
            raise ValueError("Fresh UI action lacks its POST receipt")
        submitted = json.loads(receipt["post_data"])
        if submitted["revision"] != intent["before"]["revision"]:
            raise ValueError("Browser request has incorrect starting revision")
        if intent["kind"]=="move" and [submitted["row"], submitted["col"]] != intent["move"]:
            raise ValueError("Browser request coordinate differs from challenger decision")
    image_hash = runtime.capture(screenshot, snapshot)
    evidence = {"format": FORMAT, "game_id": game_id, "intent": intent, "dom_and_page": snapshot,
                "browser_network_receipt": receipt, "recovered_after_restart": recovered,
                "screenshot": screenshot.name, "screenshot_sha256": image_hash,
                "screenshot_scope": "actual browser canvas after the atomic human click and automatic AI reply"}
    compressed = folder/(stamp+".json.gz")
    with compressed.open("xb") as stream:
        stream.write(gzip.compress(canonical_json(evidence).encode("utf-8"), mtime=0))
    journal.append("action_completed", {"game_id": game_id, "intent_seq": intent["intent_seq"],
        "kind": intent["kind"], "revision": snapshot["state"]["revision"],
        "evidence": str(compressed.relative_to(run)), "evidence_sha256": file_hash(compressed),
        "board_sha256": digest(snapshot["state"]["board"]), "recovered": recovered})
    runtime.receipts = [receipt]
    return snapshot


def perform_action(run, journal, runtime, game_id, before, kind, ai_side=None, choice=None, existing_intent=None):
    move = tuple(choice["move"]) if choice else None
    intent = existing_intent or {"game_id": game_id, "kind": kind, "before": light_state(before["state"]),
        "ai_side": ai_side, "move": list(move) if move else None, "opponent_decision": choice,
        "input_method": "CDP trusted mouse/key input"}
    if existing_intent is None:
        event = journal.append("action_intent", intent)
        intent = {**intent, "intent_seq": event["seq"]}
    current = runtime.wait_state()
    if current["state"]["revision"] == intent["before"]["revision"]+1:
        verify_transition(intent["before"], current["state"], kind, intent["move"], intent["ai_side"])
        receipt = runtime.find_receipt(current)
        deadline = time.monotonic()+5
        while receipt is None and time.monotonic()<deadline:
            time.sleep(.05)
            receipt = runtime.find_receipt(current)
        if receipt is None:
            raise RuntimeError("Restored committed action has no matching browser GET/POST receipt")
        return save_evidence(run, journal, game_id, current, receipt, intent, runtime, recovered=True)
    if current["state"]["revision"] != intent["before"]["revision"] or current["state"]["history"] != intent["before"]["history"]:
        raise ValueError("Persisted position does not match the pending browser action")
    if kind=="new":
        runtime.choose_color(intent["ai_side"])
        runtime.click_element("new")
    else:
        runtime.click_cell(*intent["move"])
    deadline = time.monotonic()+120
    while time.monotonic()<deadline:
        current = runtime.wait_state()
        if current["state"]["revision"] > intent["before"]["revision"]:
            verify_transition(intent["before"], current["state"], kind, intent["move"], intent["ai_side"])
            receipt = runtime.find_receipt(current)
            if receipt is not None:
                return save_evidence(run, journal, game_id, current, receipt, intent, runtime,
                                     recovered=existing_intent is not None)
        if current["dom"]["error"] and current["state"]["revision"] == intent["before"]["revision"]:
            raise RuntimeError("Browser rejected action: "+current["dom"]["error"])
        time.sleep(.05)
    raise TimeoutError("Click has no confirmed committed response; position is retained")


def read_latest(run, events, game_id):
    actions = [e for e in events if e["kind"]=="action_completed" and e["data"]["game_id"]==game_id]
    if not actions:
        return None
    item = actions[-1]["data"]
    path = Path(run)/item["evidence"]
    if file_hash(path) != item["evidence_sha256"]:
        raise ValueError("Browser evidence changed")
    return json.loads(gzip.decompress(path.read_bytes()))["dom_and_page"]



def synchronize_registry(run, manifest, journal):
    """Keep cross-run candidate failures append-only, including recovered completed games."""
    path = Path(manifest["registry"])
    with exclusive(path.with_suffix(".lock")):
        registry = Journal(path)
        known = {(e["data"]["run_id"], e["data"]["game_id"])
                 for e in registry.events if e["kind"] == "candidate_game"}
        for event in journal.events:
            if event["kind"] != "game_completed":
                continue
            game = event["data"]
            result_file = Path(run)/game["result_file"]
            if file_hash(result_file) != game["result_sha256"]:
                raise ValueError("Completed game result was modified")
            key = (manifest["run_id"], game["game_id"])
            if key not in known:
                registry.append("candidate_game", {**game, "run_id": manifest["run_id"],
                    "candidate_id": manifest["candidate_id"], "formal": manifest["formal"],
                    "manifest_sha256": digest(manifest)})
                known.add(key)
        return [e["data"] for e in registry.events if e["kind"] == "candidate_game"
                and e["data"]["candidate_id"] == manifest["candidate_id"]]



def registered_candidate_failures(games):
    black_losses = sum(g["ai_side"] == 1 and g["outcome"] == "loss" for g in games)
    per_run = {}
    for game in games:
        if game["formal"]:
            per_run.setdefault(game["run_id"], []).append(game)
    white_failed_runs = []
    for run_id, records in per_run.items():
        seen, sample = set(), []
        for game in records:
            if game["canonical_trajectory"] in seen:
                continue
            seen.add(game["canonical_trajectory"])
            if game["ai_side"] == 2 and len(sample) < 1000:
                sample.append(game)
        if len(sample) == 1000 and sum(g["outcome"] in ("win", "draw") for g in sample) <= 500:
            white_failed_runs.append(run_id)
    return {"black_losses": black_losses, "white_failed_runs": white_failed_runs}


def finish_game(run, manifest, journal, game_id, state):
    _, won, terminal = verify_state(state)
    if not terminal:
        raise ValueError("A truncated game cannot count as a draw")
    ai = state["ai_side"]
    outcome = "draw" if won == 0 else "win" if won == ai else "loss"
    result = {"game_id": game_id, "ai_side": ai, "winner": won, "outcome": outcome,
              "plies": len(state["history"]), "canonical_trajectory": trajectory_key(state["history"], ai),
              "termination": "five_or_more" if won else "full_board",
              "history": state["history"], "final_state": light_state(state)}
    path = Path(run)/"games"/game_id/"result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != result:
            raise ValueError("Previously saved terminal result disagrees; refusing replacement")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(result))
            stream.flush()
            os.fsync(stream.fileno())
    compact = {k: v for k, v in result.items() if k not in ("history", "final_state")}
    compact.update(result_file=path.relative_to(run).as_posix(), result_sha256=file_hash(path))
    journal.append("game_completed", compact)
    synchronize_registry(run, manifest, journal)
    return compact


def run_games(run, manifest, max_games):
    run = Path(run)
    journal = Journal(run/"events.jsonl")
    manifest_sha = digest(manifest)
    if journal.events:
        first = journal.events[0]
        if first["kind"] != "run_created" or first["data"]["manifest_sha256"] != manifest_sha:
            raise ValueError("Run manifest changed after acceptance began")
    else:
        journal.append("run_created", {"manifest_sha256": manifest_sha, "run_id": manifest["run_id"]})
    registered = synchronize_registry(run, manifest, journal)
    prior_failures = registered_candidate_failures(registered)
    historical_loss = prior_failures["black_losses"] > 0
    historical_white_failure = bool(prior_failures["white_failed_runs"])
    if historical_loss or historical_white_failure:
        status = summarize(manifest, journal.events)
        status.update(status="candidate_failed", goal_met=False,
                      candidate_historical_white_sample_failed=historical_white_failure,
                      candidate_failed_white_runs=prior_failures["white_failed_runs"],
                      candidate_historical_black_losses=sum(
                          g["ai_side"] == 1 and g["outcome"] == "loss" for g in registered))
        atomic_json(run/"run_status.json", status)
        print(canonical_json(status), flush=True)
        return 3
    runtime, failed = None, False
    try:
        atomic_json(run/"run_status.json", summarize(manifest, journal.events, running=True))
        runtime = Runtime.__new__(Runtime)
        Runtime.__init__(runtime, run, manifest, journal)
        completed_now = 0
        while completed_now < max_games:
            status = summarize(manifest, journal.events, running=True)
            atomic_json(run/"run_status.json", status)
            if status["status"] == "candidate_failed" or status["goal_met"]:
                break
            completed = {e["data"]["game_id"] for e in journal.events if e["kind"] == "game_completed"}
            starts = [e["data"] for e in journal.events if e["kind"] == "game_started"]
            active = [g for g in starts if g["game_id"] not in completed]
            if len(active) > 1:
                raise ValueError("Multiple unfinished games; refusing to choose which result survives")
            if active:
                game = active[0]
            else:
                index = len(starts)
                game = {"game_id": f"game_{index+1:06d}", "index": index,
                        "ai_side": next_ai_side(manifest, journal.events, index), "opening": "empty_board",
                        "ai_black_opening": [8, 8], "seed": manifest["seed"]}
                journal.append("game_started", game)
            game_id = game["game_id"]
            finished_intents = {e["data"]["intent_seq"] for e in journal.events
                                if e["kind"] == "action_completed" and e["data"]["game_id"] == game_id}
            intents = [e for e in journal.events if e["kind"] == "action_intent"
                       and e["data"]["game_id"] == game_id and e["seq"] not in finished_intents]
            if len(intents) > 1:
                raise ValueError("Multiple unconfirmed UI actions")
            snapshot = runtime.wait_state()
            latest = read_latest(run, journal.events, game_id)
            if intents:
                event = intents[0]
                intent = {**event["data"], "intent_seq": event["seq"]}
                snapshot = perform_action(run, journal, runtime, game_id, snapshot,
                    intent["kind"], ai_side=intent["ai_side"],
                    existing_intent=intent)
            elif latest is None:
                snapshot = perform_action(run, journal, runtime, game_id, snapshot, "new",
                                          ai_side=game["ai_side"])
            elif light_state(snapshot["state"]) != light_state(latest["state"]):
                raise ValueError("Persisted current game differs from latest confirmed browser evidence")
            while not snapshot["state"]["finished"]:
                state = snapshot["state"]
                board, _, _ = verify_state(state)
                context = {"run_id": manifest["run_id"], "game_index": game["index"],
                           "ai_side": game["ai_side"], "ply": len(state["history"]),
                           "seed": manifest["seed"], "history": state["history"],
                           "opponent_options": manifest["opponent_options"]}
                choice = runtime.opponent.request({"board": board.tolist(),
                                                    "side": state["human_side"], "context": context})
                move = choice.get("move")
                if not isinstance(move, (list, tuple)) or len(move) != 2 or any(
                        isinstance(v, bool) or not isinstance(v, int) for v in move):
                    raise ValueError("Challenger returned noninteger move")
                row, col = move
                if not 0 <= row < 16 or not 0 <= col < 16 or board[row, col] != 0:
                    raise ValueError("Challenger returned illegal move")
                snapshot = perform_action(run, journal, runtime, game_id, snapshot, "move",
                                          ai_side=game["ai_side"], choice=choice)
                if len(snapshot["state"]["history"]) % 8 in (0, 1):
                    print(canonical_json({"game": game_id, "ai_side": game["ai_side"],
                        "confirmed_plies": len(snapshot["state"]["history"]),
                        "revision": snapshot["state"]["revision"]}), flush=True)
                atomic_json(run/"run_status.json", summarize(manifest, journal.events, running=True))
            result = finish_game(run, manifest, journal, game_id, snapshot["state"])
            print(canonical_json({"completed": result}), flush=True)
            completed_now += 1
        status = summarize(manifest, journal.events)
        atomic_json(run/"run_status.json", status)
        print(canonical_json(status), flush=True)
        return 3 if status["status"] == "candidate_failed" else 0
    except BaseException as exc:
        failed = True
        journal.append("run_error", {"type": type(exc).__name__, "message": str(exc),
                       "position_retained": True})
        atomic_json(run/"run_status.json", summarize(manifest, journal.events, str(exc)))
        raise
    finally:
        if runtime is not None:
            runtime.close(keep_server=failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-callback", help=argparse.SUPPRESS)
    parser.add_argument("--worker-options-json", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--fingerprint-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", type=Path, default=ROOT/"training_runs/unet_curriculum_v2")
    parser.add_argument("--strategy", type=Path, default=ROOT/"training_runs/global_v1/global.pt")
    parser.add_argument("--opponent", default="match_challenger:choose_move")
    parser.add_argument("--opponent-options", default="{}")
    parser.add_argument("--opponent-options-file", type=Path)
    parser.add_argument("--max-games", type=int, default=2,
                        help="Maximum additional complete games this invocation; never truncates a game")
    parser.add_argument("--formal", action="store_true", help="Explicit formal run; default is a development probe")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--engine", choices=("python", "native", "guarded"), default="native")
    parser.add_argument("--search-seconds", type=float, default=.5)
    parser.add_argument("--search-nodes", type=int, default=50000)
    parser.add_argument("--search-depth", type=int, default=9)
    parser.add_argument("--search-width", type=int, default=16)
    parser.add_argument("--forcing-seconds", type=float, default=None)
    parser.add_argument("--guard-native-fraction", type=float, default=None)
    parser.add_argument("--guard-probe-nodes", type=int, default=None)
    parser.add_argument("--guard-probe-seconds", type=float, default=None)
    parser.add_argument("--guard-threat-seconds", type=float, default=None)
    parser.add_argument("--guard-threat-nodes", type=int, default=None)
    parser.add_argument("--guard-threat-width", type=int, default=None)
    parser.add_argument("--guard-threat-quiet-plies", type=int, default=None)
    parser.add_argument("--guard-threat-total-plies", type=int, default=None)
    parser.add_argument("--guard-attack-seconds", type=float, default=None)
    parser.add_argument("--guard-attack-nodes", type=int, default=None)
    parser.add_argument("--forcing-nodes", type=int, default=None)
    parser.add_argument("--forcing-depth", type=int, default=None)
    parser.add_argument("--value-weight", type=float, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args(argv)
    if args.worker_callback:
        return worker_main(args.worker_callback, json.loads(args.worker_options_json), args.fingerprint_only)
    if args.run_dir is None:
        parser.error("--run-dir is required")
    if args.max_games <= 0:
        parser.error("--max-games must be positive")
    guard_fields = ("guard_native_fraction", "guard_probe_nodes", "guard_probe_seconds",
                    "guard_threat_seconds", "guard_threat_nodes", "guard_threat_width",
                    "guard_threat_quiet_plies", "guard_threat_total_plies", "guard_attack_seconds", "guard_attack_nodes")
    if args.engine == "guarded":
        if args.forcing_seconds is not None or args.forcing_nodes is not None:
            parser.error("--forcing-seconds/nodes do not apply to guarded; use --guard-probe-*")
        for name, default in zip(guard_fields, (.45, 5000, .05, .2, 20000, 16, 2, 64, .15, 20000)):
            if getattr(args, name) is None:
                setattr(args, name, default)
        if (not np.isfinite(args.guard_native_fraction) or not 0 <= args.guard_native_fraction <= 1
                or args.guard_probe_nodes < 1 or not np.isfinite(args.guard_probe_seconds)
                or args.guard_probe_seconds <= 0 or not np.isfinite(args.guard_threat_seconds)
                or args.guard_threat_seconds < 0 or args.guard_threat_nodes < 1 or args.guard_threat_width < 1
                or not 0 <= args.guard_threat_quiet_plies <= 32 or not 1 <= args.guard_threat_total_plies <= 256
                or not np.isfinite(args.guard_attack_seconds) or args.guard_attack_seconds < 0
                or args.guard_attack_nodes < 1):
            parser.error("invalid guarded search allocation")
    else:
        if any(getattr(args, name) is not None for name in guard_fields):
            parser.error("--guard-* options require --engine guarded")
        args.forcing_seconds = .1 if args.forcing_seconds is None else args.forcing_seconds
        args.forcing_nodes = 10000 if args.forcing_nodes is None else args.forcing_nodes
    args.forcing_depth = (32 if args.engine == "guarded" else 24) if args.forcing_depth is None else args.forcing_depth
    for name in ("search_seconds", "search_nodes", "search_depth", "search_width",
                 "forcing_depth", "threads"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.forcing_seconds is not None and (not np.isfinite(args.forcing_seconds) or args.forcing_seconds < 0):
        parser.error("--forcing-seconds must be finite and nonnegative")
    if args.forcing_nodes is not None and args.forcing_nodes <= 0:
        parser.error("--forcing-nodes must be positive")
    if not np.isfinite(args.value_weight) or args.value_weight < 0:
        parser.error("--value-weight must be finite and nonnegative")
    run = args.run_dir.resolve()
    if args.resume:
        manifest = json.loads((run/"manifest.json").read_text(encoding="utf-8"))
        frozen_driver = Path(manifest["source_root"])/Path(__file__).name
        if Path(__file__).resolve() != frozen_driver.resolve():
            return subprocess.call([sys.executable, "-B", str(frozen_driver), "--run-dir", str(run),
                                    "--resume", "--max-games", str(args.max_games)],
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    else:
        manifest = freeze_run(args)
    verify_frozen(manifest)
    with exclusive(run/"runtime"/"runner.lock"):
        return run_games(run, manifest, args.max_games)


if __name__ == "__main__":
    raise SystemExit(main())
