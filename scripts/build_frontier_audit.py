from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "preserved_artifacts"
REGISTRY = ROOT / "registry" / "experiments.jsonl"
OUTPUT = ROOT / "experiments" / "post_v836_frontier.json"

TAXONOMY = [
    "named_task_family",
    "named_operator",
    "named_memory_primitive",
    "named_routing_primitive",
    "named_search_strategy",
    "named_restructuring_operation",
    "named_experiment_acquisition_heuristic",
    "explicit_recursion_depth",
    "explicit_objective",
    "explicit_noise_threshold",
    "explicit_resource_price",
    "global_backpropagation",
    "task_label_domain_label",
    "oracle_candidate_set",
    "human_defined_primitive_vocabulary",
]

KEYWORDS: dict[str, tuple[str, ...]] = {
    "named_task_family": (
        "family", "nand_native", "xor_native", "macro_native", "static_xor", "delayed_copy",
        "conditional_route", "digits", "task label", "domain label",
    ),
    "named_operator": (" add ", " sub ", " mul ", " nand ", " xor ", "operator", "dot2", "implies", "xnor"),
    "named_memory_primitive": ("memory", "write", "hold", "slot", "recurrent update", "state register"),
    "named_routing_primitive": ("router", "routing", "gate", "context routing", "route"),
    "named_search_strategy": ("search", "beam", "greedy", "multistart", "restart", "portfolio", "a*", "dijkstra"),
    "named_restructuring_operation": ("birth", "death", "prun", "split", "merge", "growth", "mutation", "duplicate", "restructure", "graph edit"),
    "named_experiment_acquisition_heuristic": ("acquisition", "information gain", "falsif", "probe", "coverage", "uncertainty", "intervention", "audit"),
    "explicit_recursion_depth": ("recursive", "recursion", "depth", "call graph", "call-graph", "hierarch", "balanced tree"),
    "explicit_objective": ("reward", "loss", "accuracy", "mse", "regret", "cost", "value", "likelihood"),
    "explicit_noise_threshold": ("noise", "corrupt", "threshold", "confidence", "lcb", "standard error", " se ", "audit"),
    "explicit_resource_price": ("resource", "cost", "budget", "compute", "regret", "energy", "evaluations", "evals"),
    "global_backpropagation": ("backprop", "gradient", "mlp", "neural network", "pytorch", "torch"),
    "task_label_domain_label": ("family", "task label", "domain label", "nand_native", "xor_native", "macro_native"),
    "oracle_candidate_set": ("oracle", "candidate", "library", "vocabulary", "genome space", "candidate set"),
    "human_defined_primitive_vocabulary": ("add/sub/mul", "add", "sub", "mul", "nand", "primitive vocabulary", "operator family", "dot2"),
}

MECHANISM_GROUPS: dict[str, tuple[str, ...]] = {
    "primitive_or_macro": ("primitive", "macro", "vocabulary", "operator", "compilation", "call graph", "call-graph"),
    "structure": ("structure", "growth", "birth", "death", "prun", "bond", "organ", "merge", "mutation", "topology"),
    "acquisition_or_experiment": ("acquisition", "probe", "falsif", "information", "uncertainty", "audit", "intervention"),
    "search": ("search", "beam", "portfolio", "greedy", "restart", "multistart", "routing"),
    "memory_or_state": ("memory", "state", "recurrent", "horizon", "retrieval"),
    "resource_control": ("budget", "resource", "cost", "compute", "regret", "stop"),
    "abstraction_or_recursion": ("abstraction", "recursive", "hierarch", "depth", "composition", "macro"),
}

SECTION_RE = re.compile(r"^##+\s+(V\d+[A-Za-z0-9]*)\s*(?:[—-]\s*)?(.*)$", re.IGNORECASE)


def normalize(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9_+*/.-]+", " ", text.lower()) + " "


def parse_markdown_sections() -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    for path in sorted(ARCHIVE.glob("*.md")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        current: dict[str, Any] | None = None
        for line in lines:
            match = SECTION_RE.match(line)
            if match:
                version = match.group(1).upper()
                current = {"version": version, "title": match.group(2).strip(), "body_lines": [], "source": str(path.relative_to(ROOT)).replace("\\", "/")}
                sections[version] = current
                continue
            if current is not None:
                if line.startswith("## ") and not SECTION_RE.match(line):
                    current = None
                else:
                    current["body_lines"].append(line)
    for section in sections.values():
        section["body"] = "\n".join(section.pop("body_lines")).strip()
    return sections


def matching_local_files(row: dict[str, Any]) -> list[Path]:
    version = row["id"].lower()
    number = str(row["number"])
    suffix = row.get("suffix", "").lower()
    prefix = f"v{number}{suffix}"
    candidates: list[Path] = []
    for rel in row.get("files", []):
        rel_path = ARCHIVE / rel
        if rel_path.exists() and rel_path.is_file():
            candidates.append(rel_path)
        parent = rel_path.parent
        if parent.exists() and parent.is_dir():
            candidates.extend(p for p in parent.iterdir() if p.is_file() and p.stem.lower().startswith(prefix))
    # Registry rows may be placeholders with no direct file. Search a narrow directory-name pattern.
    if not candidates:
        for directory in ARCHIVE.glob(f"transmutor_experiments_v{number}*plus"):
            candidates.extend(p for p in directory.iterdir() if p.is_file() and p.stem.lower().startswith(prefix))
    unique: dict[str, Path] = {}
    for path in candidates:
        unique[str(path.resolve()).lower()] = path
    return sorted(unique.values(), key=lambda p: p.name.lower())


def title_from_files(version: str, files: list[Path]) -> str:
    if not files:
        return "narrative/result artifact not separately named"
    prefix = version.lower()
    names: list[str] = []
    for path in files:
        stem = path.stem.lower()
        stem = re.sub(r"^v\d+[a-z]*_?", "", stem)
        stem = re.sub(r"_results?$", "", stem)
        if stem and stem not in names:
            names.append(stem.replace("_", " "))
    return "; ".join(names[:4]) if names else files[0].stem


def extract_status(section_body: str, json_obj: Any, version: str) -> str:
    first = next((line.strip() for line in section_body.splitlines() if line.strip()), "")
    upper = first.upper()
    if upper.startswith("PASS"):
        return "PASS"
    if upper.startswith("FAIL") or "FAIL" in upper[:40]:
        return "FAIL"
    if upper.startswith("MIXED"):
        return "MIXED"
    if "FINDING CONFIRMED" in upper:
        return "FINDING_CONFIRMED"
    if isinstance(json_obj, dict):
        bool_keys = [
            f"{version.upper()}_PASS",
            "PASS",
            "pass",
        ]
        for key in bool_keys:
            if key in json_obj and isinstance(json_obj[key], bool):
                return "PASS" if json_obj[key] else "FAIL"
        for key, value in json_obj.items():
            if key.upper().endswith("_PASS") and isinstance(value, bool):
                return "PASS" if value else "FAIL"
            if key.upper().endswith("_FINDING") and value is True:
                return "FINDING_CONFIRMED"
    return "NOT_EXPLICITLY_PRESERVED"


def load_result_json(files: list[Path]) -> tuple[Any, str | None]:
    for path in files:
        if path.suffix.lower() == ".json" and "result" in path.stem.lower():
            try:
                return json.loads(path.read_text(encoding="utf-8")), str(path.relative_to(ROOT)).replace("\\", "/")
            except Exception:
                return None, str(path.relative_to(ROOT)).replace("\\", "/")
    return None, None


def extract_lesson(body: str) -> str:
    if not body:
        return "Narrative lesson not preserved in the migrated evidence for this record."
    lines = [line.strip() for line in body.splitlines()]
    for marker in ("Conclusion:", "Lesson:", "Key result:", "Cause:", "Failure interpretation:"):
        for index, line in enumerate(lines):
            if line.lower().startswith(marker.lower()):
                collected: list[str] = []
                remainder = line[len(marker):].strip()
                if remainder:
                    collected.append(remainder)
                for nxt in lines[index + 1:index + 5]:
                    if not nxt or nxt == "---" or nxt.startswith("##"):
                        break
                    collected.append(nxt.lstrip("- "))
                if collected:
                    return " ".join(collected)
    meaningful = [line for line in lines if line and line != "---" and not line.startswith("#")]
    return " ".join(meaningful[-3:]) if meaningful else "Narrative lesson not explicitly preserved."


def extract_fix(body: str) -> str:
    for line in body.splitlines():
        clean = line.strip().lstrip("- ")
        low = clean.lower()
        if any(token in low for token in ("fix", "corrected", "restored", "repair", "removed", "separated")):
            return clean
    return "No explicit fix statement preserved for this record."


def extract_limitation(body: str) -> str:
    for line in body.splitlines():
        clean = line.strip().lstrip("- ")
        low = clean.lower()
        if any(token in low for token in ("unresolved", "failure", "failed", "caveat", "warning", "bottleneck", "not yet", "remains")):
            return clean
    return "No explicit remaining-limitation sentence preserved for this record."


def scaffold_flags(text: str) -> dict[str, str]:
    normalized = normalize(text)
    out: dict[str, str] = {}
    for key in TAXONOMY:
        needles = KEYWORDS[key]
        out[key] = "yes" if any(needle in normalized for needle in needles) else "unknown"
    # Explicit evidence that a high-level label was intentionally removed.
    if "without human labels" in normalized or "no domain label" in normalized:
        out["task_label_domain_label"] = "no"
    if "without named dist/disag" in normalized or "without named" in normalized:
        out["named_experiment_acquisition_heuristic"] = "no"
    if "neutral substrate" in normalized or "neutral cell" in normalized:
        out["named_memory_primitive"] = "no"
        out["named_routing_primitive"] = "no"
    return out


def mechanism_tags(text: str) -> list[str]:
    normalized = normalize(text)
    return [name for name, needles in MECHANISM_GROUPS.items() if any(needle in normalized for needle in needles)]


def main() -> int:
    rows = [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if 450 <= int(row["number"]) <= 836]
    sections = parse_markdown_sections()
    records: list[dict[str, Any]] = []

    for row in rows:
        version = row["id"].upper()
        section = sections.get(version, {})
        files = matching_local_files(row)
        result_obj, result_path = load_result_json(files)
        title = section.get("title") or title_from_files(version, files)
        body = section.get("body", "")
        file_paths = [str(path.relative_to(ROOT)).replace("\\", "/") for path in files]
        evidence_text_parts = [version, title, body, " ".join(file_paths)]
        if isinstance(result_obj, dict):
            evidence_text_parts.append(" ".join(map(str, result_obj.keys())))
        evidence_text = "\n".join(evidence_text_parts)

        if body and result_path:
            evidence_level = "NARRATIVE_PLUS_RESULT"
        elif body:
            evidence_level = "NARRATIVE_ONLY"
        elif result_path:
            evidence_level = "RESULT_ARTIFACT_ONLY"
        elif any(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in files):
            evidence_level = "FIGURE_ONLY"
        else:
            evidence_level = "REGISTRY_ONLY"

        fresh_text = normalize(evidence_text)
        fresh_status = "PRESERVED_FRESH_OR_AUDIT_EVIDENCE" if any(token in fresh_text for token in ("fresh", "replication", "audit")) else "NOT_EXPLICITLY_PRESERVED"

        record = {
            "version": version,
            "number": row["number"],
            "suffix": row.get("suffix", ""),
            "question": f"Preserved experiment topic: {title}" if title else "Question not explicitly preserved.",
            "mechanism": mechanism_tags(evidence_text),
            "human_supplied_scaffold": scaffold_flags(evidence_text),
            "what_was_learned": extract_lesson(body) if body else f"Artifact topic preserved as: {title}.",
            "what_was_fixed": extract_fix(body),
            "pass_fail": extract_status(body, result_obj, version),
            "fresh_audit_status": fresh_status,
            "main_lesson": extract_lesson(body),
            "remaining_limitation": extract_limitation(body),
            "evidence_level": evidence_level,
            "evidence_paths": ([section["source"]] if section else []) + file_paths,
            "registry_provenance": row.get("provenance"),
        }
        records.append(record)

    records.sort(key=lambda item: (int(item["number"]), item["suffix"]))
    if len(records) != 428:
        raise SystemExit(f"Expected 428 registry records from V450-V836, found {len(records)}")
    numeric = {int(record["number"]) for record in records}
    missing = [number for number in range(450, 837) if number not in numeric]
    if missing:
        raise SystemExit(f"Missing numeric experiment versions: {missing}")

    status_counts = Counter(record["pass_fail"] for record in records)
    evidence_counts = Counter(record["evidence_level"] for record in records)
    scaffold_yes_counts = {
        key: sum(record["human_supplied_scaffold"][key] == "yes" for record in records)
        for key in TAXONOMY
    }

    output = {
        "range": {"start": 450, "end": 836, "record_count": len(records), "numeric_version_count": len(numeric)},
        "method": {
            "rule": "Conservative evidence classification from immutable registry rows, preserved markdown sections, result JSONs, and named figures. Unknown is retained instead of guessed.",
            "warning": "A yes flag means the preserved evidence explicitly names that scaffold. Unknown does not mean absent. This matrix is an audit/index, not a claim that every historical implementation detail survived migration.",
        },
        "scaffold_taxonomy": TAXONOMY,
        "summary": {
            "status_counts": dict(status_counts),
            "evidence_counts": dict(evidence_counts),
            "explicit_scaffold_yes_counts": scaffold_yes_counts,
        },
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(records)} records / {len(numeric)} numeric versions")
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
