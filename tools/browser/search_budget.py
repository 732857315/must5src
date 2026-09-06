"""Versioned node guards for frozen browser candidates; old evidence keeps v1."""
import hashlib
import json
from pathlib import Path

LEGACY_AI_NODES = 150000
BUDGET_MODULE = "search-budget.mjs"
# This exact module is part of each new candidate's frozen web identity. The
# verifier never accepts an arbitrary node limit from a result or manifest.
V2_MODULE = '// Versioned search limits. Wall time, including inference, remains authoritative.\nexport const SEARCH_BUDGET_VERSION = 2;\nexport function searchNodeCap(seconds) {\n  return Math.min(30000000, Math.max(15000, Math.round(seconds * 1000000)));\n}\nexport function mainNodeCap(seconds) {\n  return Math.min(1500000, Math.max(5000, Math.floor(seconds * 50000)));\n}\n'
V2_SHA256 = hashlib.sha256(V2_MODULE.encode("utf-8")).hexdigest()
V2_BUDGET = dict(version=2, module=BUDGET_MODULE, sha256=V2_SHA256,
                 nodes_per_second=1000000, minimum_nodes=15000,
                 maximum_nodes=30000000, main_nodes_per_second=50000)


def frozen_search_budget(web, hashes, declaration=None, *, require_declaration=True):
    """Authenticate the frozen module and return fresh policy metadata, or v1.

    No module means historical v1 (150k at the required one second). A new
    declaration alone can never retroactively relax a historical candidate.
    """
    if BUDGET_MODULE not in hashes:
        if declaration is not None:
            raise ValueError("Search budget declaration lacks its frozen module")
        return None
    path = Path(web) / BUDGET_MODULE
    if (not path.is_file() or hashes[BUDGET_MODULE] != V2_SHA256
            or hashlib.sha256(path.read_bytes()).hexdigest() != V2_SHA256):
        raise ValueError("Unknown or changed frozen search budget module")
    if require_declaration or declaration is not None:
        # Canonical JSON also distinguishes bools and floats from required ints.
        if json.dumps(declaration, sort_keys=True) != json.dumps(V2_BUDGET, sort_keys=True):
            raise ValueError("Frozen search budget declaration differs from its module")
    return dict(V2_BUDGET)


def acceptance_node_limit(budget):
    # All acceptance games independently require exactly one second.
    return LEGACY_AI_NODES if budget is None else budget["nodes_per_second"]
