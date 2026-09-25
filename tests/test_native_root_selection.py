"""Exercise the real native root loop with deterministic child-result fixtures.

Only recursive search is substituted in a temporary translation unit. The
production ranking, node checks, move undo and proof aggregation stay intact;
there is no timing-based search or training in these tests.
"""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


MATE = 100000000
UNKNOWN = 2


class NativeOutput(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in
                ("move", "nodes", "completed_depth", "score", "proof",
                 "budget_exhausted", "status")]


FIXTURE_SEARCH = """
static Result fixture_results[MAX_CELLS];
EXPORT void fixture_result(int move,int score,int proof) {
    fixture_results[move].score=score;
    fixture_results[move].proof=proof;
}
EXPORT int fixture_board_at(void *context,int point) {
    return ((Context*)context)->board[point];
}
EXPORT void fixture_set_point(void *context,int point,int value) {
    set_point((Context*)context,point,value);
}
EXPORT int fixture_verify_incremental(void *context) {
    Context *c=(Context*)context;
    static Work expected,actual;
    analyse_full(c,&expected);analyse(c,&actual);
    if(expected.empty_count!=actual.empty_count) return 1;
    u64 hash=piece_key(c->rows*65+c->cols,0);
    for(int i=0;i<c->cells;i++) if(c->board[i]) hash^=piece_key(i,c->board[i]);
    if(hash!=c->hash) return 2;
    for(int s=0;s<2;s++) {
        if(expected.evaluation[s]!=actual.evaluation[s]||expected.win_count[s]!=actual.win_count[s]) return 3;
        for(int i=0;i<c->cells;i++)
            if(expected.score[s][i]!=actual.score[s][i]||expected.wins[s][i]!=actual.wins[s][i]||
               expected.upgrades[s][i]!=actual.upgrades[s][i]) return 4;
    }
    for(int i=0;i<c->cells;i++) if(expected.frontier[i]!=actual.frontier[i]) return 5;
    return 0;
}
static Result search(Context *c,int side,int depth,int alpha,int beta,int ply,int last,int quiescence) {
    Result result={0,UNKNOWN};
    if(!check(c)) return result;
    c->nodes++;
    return fixture_results[last];
}
"""


class NativeRootSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clang, linker = shutil.which("clang"), shutil.which("lld-link")
        if os.name != "nt" or not clang or not linker:
            raise unittest.SkipTest("Native root fixture requires Windows clang and lld-link")
        cls.temporary = tempfile.TemporaryDirectory(prefix="must5-native-root-")
        try:
            directory = Path(cls.temporary.name)
            production = (Path(__file__).resolve().parents[1] / "native_board.c").read_text(encoding="utf8")
            start = production.index("static Result search(")
            end = production.index("EXPORT u64 native_context_size", start)
            # Preserve native_select byte for byte, including the patched tie
            # handling and the independent complete/known proof calculation.
            fixture = production[:start] + FIXTURE_SEARCH + production[end:]
            source, obj, binary = (directory / name for name in ("fixture.c", "fixture.obj", "fixture.dll"))
            source.write_text(fixture, encoding="utf8")
            subprocess.run([clang, "-c", str(source), "-o", str(obj), "-O2",
                            "-ffreestanding", "-fno-builtin"], check=True,
                           capture_output=True, text=True, timeout=30)
            subprocess.run([linker, "/dll", "/noentry", "/nodefaultlib",
                            "/out:" + str(binary), str(obj)], check=True,
                           capture_output=True, text=True, timeout=30)
            cls.library = ctypes.CDLL(str(binary))
            cls.library.native_context_size.argtypes = []
            cls.library.native_context_size.restype = ctypes.c_ulonglong
            cls.library.fixture_result.argtypes = [ctypes.c_int] * 3
            cls.library.fixture_result.restype = None
            cls.library.fixture_board_at.argtypes = [ctypes.c_void_p, ctypes.c_int]
            cls.library.fixture_board_at.restype = ctypes.c_int
            cls.library.fixture_set_point.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            cls.library.fixture_set_point.restype = None
            cls.library.fixture_verify_incremental.argtypes = [ctypes.c_void_p]
            cls.library.fixture_verify_incremental.restype = ctypes.c_int
            cls.library.native_select.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                ctypes.POINTER(NativeOutput)]
            cls.library.native_select.restype = ctypes.c_int
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        # The DLL is only this class's temporary fixture, never a production
        # library or live server module. Release it before Windows cleanup.
        import _ctypes
        _ctypes.FreeLibrary(cls.library._handle)
        del cls.library
        cls.temporary.cleanup()

    def choose(self, first, second, *, width=2, depth=1, max_nodes=100, side=1):
        # Both cells are legal and no line is long enough to form five. The
        # geometric center is forbidden, so production frontier covers both.
        board = (ctypes.c_ubyte * 4)(0, 3, 3, 0)
        priors = (ctypes.c_double * 4)(1, 0, 0, 0)
        context = ctypes.create_string_buffer(self.library.native_context_size())
        self.library.fixture_result(0, *first)
        self.library.fixture_result(3, *second)
        result = NativeOutput()
        code = self.library.native_select(context, board, 1, 4, side, priors,
                                          depth, width, max_nodes, None,
                                          ctypes.byref(result))
        self.assertEqual(code, 0)
        self.assertEqual(list(board), [0, 3, 3, 0])
        self.assertEqual([self.library.fixture_board_at(context, i) for i in range(4)],
                         [0, 3, 3, 0])
        self.assertLessEqual(result.nodes, max_nodes)
        return result

    def test_equal_mate_prefers_unknown_over_proved_loss_for_both_colors(self):
        for side in (1, 2):
            with self.subTest(side=side):
                result = self.choose((MATE, 1), (MATE, UNKNOWN), side=side)
                self.assertEqual(result.move, 3)
                self.assertEqual(result.score, -MATE)
                self.assertEqual(result.proof, UNKNOWN)
                self.assertEqual(result.completed_depth, 1)
                self.assertEqual(result.nodes, 2)

    def test_later_proved_loss_does_not_replace_equal_unknown(self):
        result = self.choose((MATE, UNKNOWN), (MATE, 1))
        self.assertEqual(result.move, 0)
        self.assertEqual(result.proof, UNKNOWN)

    def test_two_unknown_ties_keep_existing_order(self):
        result = self.choose((MATE, UNKNOWN), (MATE, UNKNOWN))
        self.assertEqual(result.move, 0)
        self.assertEqual(result.proof, UNKNOWN)

    def test_all_legal_proved_losses_still_prove_parent_loss(self):
        result = self.choose((MATE, 1), (MATE, 1))
        self.assertEqual(result.move, 0)
        self.assertEqual(result.score, -MATE)
        self.assertEqual(result.proof, -1)
        self.assertEqual(result.completed_depth, 1)

    def test_pruned_legal_set_cannot_prove_parent_loss(self):
        result = self.choose((MATE, 1), (MATE, UNKNOWN), width=1)
        self.assertEqual(result.move, 0)
        self.assertEqual(result.proof, UNKNOWN)
        self.assertEqual(result.nodes, 1)

    def test_proved_child_loss_still_proves_parent_win(self):
        result = self.choose((MATE, 1), (-MATE, -1))
        self.assertEqual(result.move, 3)
        self.assertEqual(result.proof, 1)
        self.assertEqual(result.score, MATE)

    def test_complete_known_set_with_draw_still_proves_parent_draw(self):
        result = self.choose((MATE, 1), (0, 0))
        self.assertEqual(result.move, 3)
        self.assertEqual(result.proof, 0)
        self.assertEqual(result.score, 0)

    def test_known_draw_does_not_hide_an_unsearched_unknown(self):
        result = self.choose((0, 0), (1, UNKNOWN))
        self.assertEqual(result.move, 0)
        self.assertEqual(result.proof, UNKNOWN)

    def test_score_remains_primary_for_two_unknown_actions(self):
        result = self.choose((-17, UNKNOWN), (-23, UNKNOWN))
        self.assertEqual(result.move, 3)
        self.assertEqual(result.score, 23)
        self.assertEqual(result.proof, UNKNOWN)

    def test_incomplete_iteration_keeps_fallback_and_unknown_proof(self):
        result = self.choose((MATE, 1), (MATE, UNKNOWN), max_nodes=1)
        self.assertEqual(result.move, 0)
        self.assertEqual(result.proof, UNKNOWN)
        self.assertEqual(result.completed_depth, 0)
        self.assertEqual(result.nodes, 1)
        self.assertTrue(result.budget_exhausted)

    def test_iterative_preferred_move_does_not_restore_a_proved_losing_tie(self):
        result = self.choose((MATE, 1), (MATE, UNKNOWN), depth=3)
        self.assertEqual(result.move, 3)
        self.assertEqual(result.proof, UNKNOWN)
        self.assertEqual(result.completed_depth, 3)
        self.assertEqual(result.nodes, 6)

    def test_real_immediate_win_precedes_any_fixture_score(self):
        context = ctypes.create_string_buffer(self.library.native_context_size())
        board = (ctypes.c_ubyte * 6)(3, 1, 1, 1, 1, 0)
        priors = (ctypes.c_double * 6)()
        result = NativeOutput()
        code = self.library.native_select(context, board, 1, 6, 1, priors,
                                          1, 2, 0, None, ctypes.byref(result))
        self.assertEqual(code, 0)
        self.assertEqual(result.move, 5)
        self.assertEqual(result.proof, 1)
        self.assertEqual(result.nodes, 0)

    def test_incremental_patterns_and_hash_match_full_scan_after_moves_and_undo(self):
        import random
        rng = random.Random(20260926)
        context = ctypes.create_string_buffer(self.library.native_context_size())
        for rows, cols in ((5, 5), (8, 12), (15, 15), (32, 32), (64, 64)):
            cells = rows * cols
            board = (ctypes.c_ubyte * cells)(*(3 if rng.random() < .12 else 0 for _ in range(cells)))
            original = list(board)
            priors = (ctypes.c_double * cells)()
            output = NativeOutput()
            self.assertEqual(self.library.native_select(context, board, rows, cols, 1, priors,
                1, 16, 0, None, ctypes.byref(output)), 0)
            self.assertEqual(self.library.fixture_verify_incremental(context), 0)
            moves = [i for i in range(cells) if not board[i]]
            rng.shuffle(moves)
            moves = moves[:256]
            for i, point in enumerate(moves):
                self.library.fixture_set_point(context, point, 1 + i % 2)
                self.assertEqual(self.library.fixture_verify_incremental(context), 0, (rows, cols, i, 'play'))
            for point in reversed(moves):
                self.library.fixture_set_point(context, point, 0)
                self.assertEqual(self.library.fixture_verify_incremental(context), 0, (rows, cols, point, 'undo'))
            self.assertEqual([self.library.fixture_board_at(context, i) for i in range(cells)], original)


if __name__ == "__main__":
    unittest.main()
