"""Read validated per-writing Yes/No features from the private frozen Qwen cache."""
from contextlib import closing
from collections import Counter
import sqlite3

import final_indirect_pipeline as lexical
from llm_signal_scorer import canonical_hash, validated_cohort_signals


def verified_source(config, gate, heldout, preflight):
    signals = validated_cohort_signals(config, gate)
    prompt_hash = canonical_hash(config)
    if (heldout.get("split") != "heldout" or heldout.get("prompt_version") != 1
            or heldout.get("prompt_hash") != prompt_hash):
        raise ValueError("Matching independent held-out signal evaluation required")
    for signal in signals:
        result = heldout.get("signals", {}).get(signal, {})
        if (result.get("n") != 60 or result.get("probes_passed") is not True
                or result.get("numeric_gate_passed") is not True
                or not isinstance(result.get("model_auroc"), (float, int))
                or not isinstance(result.get("word_list_auroc"), (float, int))
                or not 0.80 <= result["model_auroc"] <= 1
                or not 0 <= result["word_list_auroc"] < result["model_auroc"]):
            raise ValueError("An admitted signal did not pass the held-out comparison")
    if (preflight.get("prompt_hash") != prompt_hash or preflight.get("model_digest") != config["expected_digest"]
            or preflight.get("phase1_passed") is not True or preflight.get("status") != "complete"
            or not preflight.get("ollama_version")):
        raise ValueError("Matching successful frozen-model preflight required")
    return {"signals": list(signals), "prompt_hash": prompt_hash, "prompt_version": 1,
            "model_digest": config["expected_digest"], "ollama_version": preflight["ollama_version"]}


class CachedPostFlags:
    """Identical scorer inputs may reuse the same categorical answer across posts."""
    def __init__(self, path, source):
        self.signals = source["signals"]
        self.answers = {}
        self.quality_counts = Counter()
        placeholders = ",".join("?" for _ in self.signals)
        query = ("SELECT text_hash, signal, score, missing_flag, truncated_flag FROM scores WHERE prompt_hash=? AND prompt_version=? "
                 "AND model_digest=? AND ollama_version=? AND signal IN (" + placeholders + ")")
        params = [source[k] for k in ("prompt_hash", "prompt_version", "model_digest", "ollama_version")]
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            for text_hash, signal, score, missing, truncated in db.execute(query, params + self.signals):
                if not isinstance(score, (float, int)) or not 0 <= score <= 1:
                    raise ValueError("Invalid frozen score in private cache")
                if missing not in (0, 1) or truncated not in (0, 1):
                    raise ValueError("Invalid frozen quality flags in private cache")
                values = self.answers.setdefault(text_hash, {})
                flag = int(score >= 0.5)
                if signal in values and values[signal][0] != flag and not any(values[signal][1:]) and not (missing or truncated):
                    raise ValueError("Identical frozen inputs have conflicting Yes/No answers")
                if signal in values:
                    old_flag, old_missing, old_truncated = values[signal]
                    values[signal] = (old_flag, old_missing or missing, old_truncated or truncated)
                else:
                    values[signal] = (flag, missing, truncated)

    def __call__(self, text):
        values = self.answers.get(canonical_hash(text), {})
        if len(values) != len(self.signals):
            raise ValueError("Missing approved Qwen answers for eligible writing. Complete the private cache first.")
        result = lexical.flags(text)
        for signal in self.signals:
            flag, missing, truncated = values[signal]
            if missing or truncated:
                if missing:
                    self.quality_counts["word_rule_fallback_missing_alternative"] += 1
                if truncated:
                    self.quality_counts["word_rule_fallback_truncated"] += 1
            else:
                result["helpless" if signal == "helplessness" else signal] = flag
                self.quality_counts["qwen_used"] += 1
        return result
