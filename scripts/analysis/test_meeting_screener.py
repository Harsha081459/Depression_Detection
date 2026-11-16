import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_signal_scorer as qwen
import meeting_qwen_features as bridge
import meeting_screener as scr
import meeting_windows as mw


class CountModel:
    def predict_proba(self, rows):
        prob = np.where(rows.frequency >= 3, .9, .1)
        return np.column_stack((1 - prob, prob))


def simple_history(days, text="I feel alone"):
    return mw.timed_events(SimpleNamespace(date=pd.Timestamp("2025-01-01", tz="UTC") + d * mw.DAY,
                                          text=text, disclosure=False) for d in days)


class ScreenerTests(unittest.TestCase):
    def test_threshold_counts_every_control_and_excludes_ties(self):
        self.assertEqual(scr.calibration_threshold([.9, .7] + [np.nan] * 8), np.nextafter(.7, np.inf))
        self.assertGreater(scr.calibration_threshold([.8] * 11 + [.2] * 89), .8)
        self.assertEqual(scr.calibration_threshold([np.nan] * 10), .5)
        self.assertEqual(scr.calibration_threshold([.9] + [np.nan] * 9), .5)

    def test_native_target_filter_and_cutoff_happen_before_extraction(self):
        def node(day, target, text):
            return {"date": "2025-01-%02d" % day, "type": "comment", "target": target,
                    "user_id": "original_writer" if target else "another_writer", "body": text}
        data = [{"comments": [node(1, False, "context"), node(2, True, "first"),
                              node(5, True, "I was diagnosed with depression"), node(5, True, "tied"),
                              node(6, True, "later"), node(7, True, "None"),
                              node(14, True, "future")]}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.json"
            path.write_text(json.dumps(data))
            seen = []
            def spy(text):
                seen.append(text)
                return mw.lexical.flags(text)
            uid, _ = scr.read_history(path, 8, spy, "before_disclosure")
            self.assertEqual(uid, "original_writer")
            self.assertEqual(seen, ["first"])
            seen.clear()
            scr.read_history(path, 8, spy, "masked")
            self.assertEqual(seen, ["first", "tied", "later"])

    def test_one_current_feature_changes_its_own_direction_only(self):
        state = list(mw.window_states(simple_history([0, 6]), 4, "masked", 8))[-1]
        base = mw.encode_levels(state["levels"], state["previous"])
        choices = list(scr.alternatives(state))
        for name, value, row in choices:
            diff = {k for k in row if row[k] != base[k]}
            self.assertIn(name, diff)
            self.assertLessEqual(diff, {name, "change_" + name})
            self.assertEqual(row["change_" + name], mw.direction(value, state["previous"][name]))
        self.assertFalse(any(name == "closed_silence" for name, _, _ in choices))
        self.assertFalse(any(name == "isolation" and value == 1 for name, value, _ in choices))

    def test_first_flag_survives_replay_and_stable_can_wait(self):
        # Three writings in the second window, including two empty ones.
        events = simple_history([0, 5, 6, 7, 9, 10, 11, 12])
        for idx in (2, 3):
            events[idx]["flags"] = []
        fold_models = [{"model": CountModel(), "threshold": .8} for _ in range(5)]
        bundle = {"source": "word_rules", "models": {"4d_masked": fold_models}, "user_folds": {"a": 2}}
        first = scr.screen(bundle, "a", events, 8, 4, "masked")
        self.assertEqual(first["model_partition"], "held_out_user")
        self.assertEqual(first["first_flag"]["day"], 8)
        self.assertEqual(first["timeline"][-1]["sensitive_features"], ["frequency"])
        stable = scr.screen(bundle, "a", events, 12, 4, "masked", "stable")
        self.assertEqual(stable["timeline"][1]["status"], "waiting_for_stable_evidence")
        self.assertEqual(stable["first_flag"]["day"], 12)
        self.assertEqual(stable["first_flag"]["single_feature_reversals"], [])
        self.assertEqual(scr.screen(bundle, "a", events, 7, 4, "masked")["decision"], "no_flag")
        self.assertEqual(scr.screen(bundle, "new", [], 8, 4, "masked")["observation"], "no_target_writings")
        with tempfile.TemporaryDirectory() as tmp:
            bundle.update(version=scr.VERSION, sklearn=scr.sklearn.__version__)
            path = Path(tmp) / "model.joblib"
            scr.save_bundle(bundle, path)
            self.assertEqual(first, scr.screen(scr.load_bundle(path), "a", events, 8, 4, "masked"))

    def test_evaluation_folds_keep_calibration_and_test_users_out_of_fit(self):
        labels = pd.Series({str(i): i % 2 for i in range(20)})
        folds = pd.Series({uid: int(uid) // 4 for uid in labels.index})
        histories = {uid: simple_history([0, 5, 6]) for uid in labels.index}
        with patch.object(mw, "fit_classifier", wraps=mw.fit_classifier) as fit:
            models, summary, _ = scr.train_schedule(histories, labels, folds, 4, "masked")
        for fold, call in enumerate(fit.call_args_list):
            used = set(call.args[0].user_id)
            self.assertEqual(used, set(folds[~folds.isin([fold, (fold+1) % 5])].index))
            self.assertEqual(models[fold]["calibration_controls"], 2)
        self.assertEqual(summary["calibrated"]["8"]["positive_assessed"], 10)
        self.assertEqual(set(summary["replay_digests"]["calibrated"]),
                         {str(day) for day in range(0, mw.HORIZON + 1, 4)})
        self.assertEqual(summary["replay_digests"]["calibrated"]["8"],
                         summary["calibrated"]["8"]["replay_digest"])

    def test_paired_schedule_counts_use_same_assessed_people(self):
        labels = pd.Series({"p1": 1, "p2": 1, "c1": 0, "c2": 0})
        short = pd.DataFrame([("p1", True), ("p2", True), ("c1", False)],
                             columns=["user_id", "calibrated"])
        long = pd.DataFrame([("p1", False), ("c1", True), ("c2", True)],
                            columns=["user_id", "calibrated"])
        counts = scr.paired_window_counts(short, long, labels, "calibrated")
        self.assertEqual(counts["positive"], {
            "both_assessed": 1, "only_4d_assessed": 1, "only_28d_assessed": 0,
            "neither_assessed": 0, "both_flagged": 0, "only_4d_flagged": 1,
            "only_28d_flagged": 0, "neither_flagged_among_both_assessed": 0})
        self.assertEqual(counts["control"]["only_28d_flagged"], 1)


class CachedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = qwen.load_prompts(qwen.HERE / "prompts.json")
        self.config["prompt_version"] = 1
        prompt_hash = qwen.canonical_hash(self.config)
        self.gate = {"prompt_hash": prompt_hash, "validated_signals": ["helplessness"]}
        self.preflight = {"prompt_hash": prompt_hash, "model_digest": self.config["expected_digest"],
                          "ollama_version": "test", "phase1_passed": True, "status": "complete"}
        self.heldout = {"split": "heldout", "prompt_version": 1, "prompt_hash": prompt_hash,
                        "signals": {"helplessness": {"n": 60, "probes_passed": True, "numeric_gate_passed": True,
                                                    "model_auroc": .9, "word_list_auroc": .8}}}
        self.result = {"score": .5, "l_yes": -1., "l_no": -1., "missing_flag": 0,
                       "truncated_flag": 0, "latency_seconds": .1}

    def test_only_passing_heldout_signals_can_enter(self):
        for bad in (dict(self.heldout, split="development"), dict(self.heldout, prompt_hash="stale"),
                    dict(self.heldout, signals={})):
            with self.assertRaises(ValueError):
                bridge.verified_source(self.config, self.gate, bad, self.preflight)
        for invalid in ([], ["helplessness", "helplessness"], "helplessness", ["other"]):
            with self.assertRaises(qwen.SafeError):
                qwen.validated_cohort_signals(self.config, dict(self.gate, validated_signals=invalid))

    def test_cached_yes_no_overrides_only_the_approved_feature(self):
        source = bridge.verified_source(self.config, self.gate, self.heldout, self.preflight)
        text = "I feel alone"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.sqlite"
            cache = qwen.ScoreCache(path)
            post = {"post_id": "one", "text": text}
            cache.put(post, "helplessness", self.config, self.preflight, self.result)
            cache.close()
            flags = bridge.CachedPostFlags(path, source)
            self.assertEqual(flags(text), dict(mw.lexical.flags(text), helpless=1))
            with self.assertRaisesRegex(ValueError, "Missing approved"):
                flags("Unscored synthetic writing")
            events = mw.timed_events([SimpleNamespace(date=pd.Timestamp("2025-01-01"), text=text,
                                                      disclosure=False)], flags)
            self.assertEqual(mw.window_levels(events)[0]["helpless"], 2)
            stale = bridge.CachedPostFlags(path, dict(source, ollama_version="old"))
            with self.assertRaisesRegex(ValueError, "Missing approved"):
                stale(text)
            cache = qwen.ScoreCache(path)
            cache.put(dict(post, post_id="two"), "helplessness", self.config, self.preflight,
                      dict(self.result, score=.1))
            cache.close()
            with self.assertRaisesRegex(ValueError, "conflicting"):
                bridge.CachedPostFlags(path, source)

    def test_incomplete_or_truncated_answers_use_explicit_word_rule_fallback(self):
        source = bridge.verified_source(self.config, self.gate, self.heldout, self.preflight)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.sqlite"
            for name, incomplete in (("missing_flag", 1), ("truncated_flag", 1)):
                cache = qwen.ScoreCache(path)
                cache.put({"post_id": name, "text": name}, "helplessness", self.config,
                          self.preflight, dict(self.result, **{name: incomplete}))
                cache.close()
                flags = bridge.CachedPostFlags(path, source)
                self.assertEqual(flags(name), mw.lexical.flags(name))
                self.assertEqual(flags.quality_counts[
                    "word_rule_fallback_missing_alternative" if name == "missing_flag"
                    else "word_rule_fallback_truncated"], 1)

    def test_cohort_runner_requests_only_the_approved_signal(self):
        seen = []
        def fake_call(text, signal, config):
            seen.append(signal)
            return self.result
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "posts.jsonl"
            path.write_text(json.dumps({"post_id": "one", "text": "Synthetic writing"}) + "\n")
            with patch.object(qwen, "os", SimpleNamespace(name="posix")), \
                 patch.object(qwen, "model_info", return_value=self.preflight), \
                 patch.object(qwen, "validate_gate"), patch.object(qwen, "score_one", fake_call):
                result = qwen.score_file(path, Path(tmp) / "scores.sqlite", 1, self.config, self.preflight,
                                         purpose="cohort", pilot_gate=self.gate, heldout_report=self.heldout)
            self.assertEqual(seen, ["helplessness"])
            self.assertEqual(result["completed"], 1)
            with self.assertRaises(qwen.SafeError):
                qwen.score_file(path, Path(tmp) / "scores.sqlite", 1, self.config, self.preflight,
                                purpose="cohort", pilot_gate=self.gate)

    def test_qwen_input_matches_locked_id_and_removes_ineligible_text(self):
        ts = pd.Timestamp("2025-01-01", tz="UTC")
        data = pd.DataFrame({"user_id": ["x"] * 4, "date": [ts] * 4,
                             "text": ["eligible", "[deleted]", "disclosure", "too late"],
                             "within_12w": [True, True, True, False],
                             "disclosure": [False, False, True, False]}, index=[12, 13, 14, 15])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.jsonl"
            self.assertEqual(scr.write_qwen_inputs(data, {"signals": ["isolation"]}, path), 1)
            out = json.loads(path.read_text())
            self.assertEqual(out, {"post_id": qwen.canonical_hash(["x", ts.isoformat(), 12, "eligible"]),
                                   "text": "eligible", "requested_signals": ["isolation"]})


if __name__ == "__main__":
    unittest.main()
