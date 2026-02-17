#!/usr/bin/env python3
"""
Check Hugging Face model access for model IDs used in this repository.

This script:
1) Scans Python files in the repo for likely HF model IDs.
2) Includes model IDs defined in `scripts/generate_compositional_model_grid.py`.
3) Uses `huggingface_hub` + your HF token to verify access to each model.

Usage:
    HF_TOKEN=hf_... python scripts/check_hf_model_access.py
    HF_TOKEN=hf_... python scripts/check_hf_model_access.py --show-sources
    HF_TOKEN=hf_... python scripts/check_hf_model_access.py --json-out results/hf_access.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCAN_DIRS = ("scripts", "notebooks", "run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check HF access for model IDs in this repo.")
    p.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repo root to scan.",
    )
    p.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF token (or use HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars).",
    )
    p.add_argument(
        "--show-sources",
        action="store_true",
        help="Print file:line sources where each model ID was found.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any model is not accessible.",
    )
    return p.parse_args()


def is_candidate_hf_model_id(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if value.count("/") != 1:
        return False
    if value.startswith(("/", ".", "http://", "https://")):
        return False
    org, repo = value.split("/", 1)
    if not org or not repo:
        return False
    if " " in value or "\\" in value:
        return False
    return True


def canonical_model_id(model_id: str) -> str:
    aliases = {
        "stabilityai/stable-diffusion-2-1": "stabilityai/stable-diffusion-2-1-base",
        "runwayml/stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    }
    return aliases.get(model_id, model_id)


def add_model(
    model_id: str,
    source: str,
    models_to_sources: Dict[str, Set[str]],
):
    if not is_candidate_hf_model_id(model_id):
        return
    models_to_sources.setdefault(model_id, set()).add(source)


class HFModelExtractor(ast.NodeVisitor):
    def __init__(self, file_path: Path, models_to_sources: Dict[str, Set[str]]):
        self.file_path = file_path
        self.models_to_sources = models_to_sources

    def _src(self, lineno: int) -> str:
        return f"{self.file_path}:{lineno}"

    def _const_str(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _extract_str_list(self, node) -> List[str]:
        out = []
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                s = self._const_str(elt)
                if s is not None:
                    out.append(s)
        return out

    def visit_Assign(self, node: ast.Assign):
        # Example: SMALL_MODEL_IDS = ["org/repo", ...]
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if any(n.endswith("_MODEL_IDS") for n in names):
            for s in self._extract_str_list(node.value):
                add_model(s, self._src(node.lineno), self.models_to_sources)

        # Example: DEFAULT_MODEL_ID = "org/repo"
        if any(n.endswith("MODEL_ID") for n in names):
            s = self._const_str(node.value)
            if s is not None:
                add_model(s, self._src(node.lineno), self.models_to_sources)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # from_pretrained("org/repo")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
            if node.args:
                s = self._const_str(node.args[0])
                if s is not None:
                    add_model(s, self._src(node.lineno), self.models_to_sources)

        # load_ip_adapter("org/repo")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "load_ip_adapter":
            if node.args:
                s = self._const_str(node.args[0])
                if s is not None:
                    add_model(s, self._src(node.lineno), self.models_to_sources)

        # parser.add_argument("--model_id", default="org/repo")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            arg_name = None
            if node.args:
                arg_name = self._const_str(node.args[0])
            if isinstance(arg_name, str) and (
                "model_id" in arg_name or "adapter_id" in arg_name or "repo_id" in arg_name
            ):
                for kw in node.keywords:
                    if kw.arg == "default":
                        s = self._const_str(kw.value)
                        if s is not None:
                            add_model(s, self._src(node.lineno), self.models_to_sources)

        self.generic_visit(node)


def scan_repo_for_model_ids(root: Path) -> Dict[str, Set[str]]:
    models_to_sources: Dict[str, Set[str]] = {}

    for d in DEFAULT_SCAN_DIRS:
        base = root / d
        if not base.exists():
            continue
        for py_file in base.rglob("*.py"):
            try:
                text = py_file.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except Exception:
                continue
            extractor = HFModelExtractor(py_file, models_to_sources)
            extractor.visit(tree)

    # Canonicalize stale IDs while keeping source traceability.
    canon: Dict[str, Set[str]] = {}
    for mid, srcs in models_to_sources.items():
        cmid = canonical_model_id(mid)
        merged = set(srcs)
        if cmid != mid:
            merged.add(f"canonicalized:{mid}->{cmid}")
        canon.setdefault(cmid, set()).update(merged)
    return canon


def check_model_access(model_ids: List[str], token: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    # Delayed imports so script can still be parsed without dependency installed.
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi(token=token)
    status: Dict[str, str] = {}
    detail: Dict[str, str] = {}

    for model_id in model_ids:
        try:
            api.model_info(model_id)
            status[model_id] = "OK"
            detail[model_id] = "accessible"
        except HfHubHTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 401:
                status[model_id] = "AUTH"
                detail[model_id] = "Unauthorized (token missing/invalid or terms not accepted)"
            elif code == 404:
                status[model_id] = "MISSING"
                detail[model_id] = "Not found or inaccessible"
            elif code == 403:
                status[model_id] = "FORBIDDEN"
                detail[model_id] = "Forbidden/gated terms not accepted"
            else:
                status[model_id] = "ERROR"
                detail[model_id] = f"HTTP {code}: {e}"
        except Exception as e:
            status[model_id] = "ERROR"
            detail[model_id] = f"{type(e).__name__}: {e}"
    return status, detail


def main():
    args = parse_args()

    if not args.token:
        raise SystemExit(
            "No HF token found. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) and rerun."
        )

    models_to_sources = scan_repo_for_model_ids(args.root)
    model_ids = sorted(models_to_sources.keys())

    if not model_ids:
        raise SystemExit("No candidate HF model IDs found in repo scan.")

    print(f"Found {len(model_ids)} model IDs. Checking access...\n")
    status, detail = check_model_access(model_ids=model_ids, token=args.token)

    ok = 0
    bad = 0
    rows = []
    for model_id in model_ids:
        st = status.get(model_id, "ERROR")
        dt = detail.get(model_id, "")
        rows.append((model_id, st, dt))
        if st == "OK":
            ok += 1
        else:
            bad += 1

    for model_id, st, dt in rows:
        print(f"[{st:9}] {model_id} :: {dt}")
        if args.show_sources:
            for src in sorted(models_to_sources.get(model_id, set())):
                print(f"           - {src}")

    print("\nSummary")
    print(f"  total : {len(model_ids)}")
    print(f"  ok    : {ok}")
    print(f"  fail  : {bad}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total": len(model_ids),
            "ok": ok,
            "fail": bad,
            "models": [
                {
                    "model_id": m,
                    "status": status.get(m, "ERROR"),
                    "detail": detail.get(m, ""),
                    "sources": sorted(models_to_sources.get(m, set())),
                }
                for m in model_ids
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  json  : {args.json_out}")

    if args.strict and bad > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
