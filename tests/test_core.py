import unittest
import _path  # noqa: F401
import numpy as np
import pandas as pd

from energy_detective.analytics.baseline import cv_rmse, nmbe
from energy_detective.analytics.detector import incidents_from_flags
from energy_detective.agent.planner import parse_query
from energy_detective.data.quality import quality_report
from energy_detective.data.simulator import generate_dataset
from energy_detective.diagnosis.rules import diagnose, incident_features
from energy_detective.knowledge.retriever import KnowledgeBase


class TestMetrics(unittest.TestCase):
    def test_cv_rmse_nmbe(self):
        y = np.array([100.0, 100.0, 100.0, 100.0])
        yhat = np.array([110.0, 90.0, 110.0, 90.0])
        self.assertAlmostEqual(cv_rmse(y, yhat), 10.0)
        self.assertAlmostEqual(nmbe(y, yhat), 0.0)
        self.assertAlmostEqual(nmbe(y, y * 1.05), 5.0)


class TestIncidents(unittest.TestCase):
    def setUp(self):
        self.idx = pd.date_range("2025-01-01", periods=100, freq="h")

    def test_min_run_and_fill_gap(self):
        f = np.zeros(100, bool)
        f[10:13] = True       # 3h run
        f[20:22] = True       # 2h run (dropped)
        f[30:33] = True
        f[34:37] = True       # gap of 1 -> joined
        incs = incidents_from_flags(f, self.idx, min_run=3, fill_gap=1, merge_gap=0)
        spans = [(i.i0, i.i1) for i in incs]
        self.assertEqual(spans, [(10, 12), (30, 36)])

    def test_strong_keeps_short_run(self):
        f = np.zeros(100, bool); f[50] = True
        s = np.zeros(100, bool); s[50] = True
        self.assertEqual(len(incidents_from_flags(f, self.idx, min_run=3, strong=s)), 1)
        self.assertEqual(len(incidents_from_flags(f, self.idx, min_run=3)), 0)

    def test_merge_gap(self):
        f = np.zeros(100, bool); f[10:14] = True; f[20:24] = True
        self.assertEqual(len(incidents_from_flags(f, self.idx, merge_gap=18)), 1)
        self.assertEqual(len(incidents_from_flags(f, self.idx, merge_gap=2)), 2)


class TestDiagnosis(unittest.TestCase):
    def _feat(self, start, n, extra, pred=100.0, temp=30.0, flat=False, zero=False):
        idx = pd.date_range(start, periods=n + 2, freq="h")
        pred_a = np.full(n + 2, pred)
        y = pred_a + np.pad(extra, (1, 1)) if hasattr(extra, "__len__") else pred_a + np.pad(np.full(n, extra), (1, 1))
        if flat:
            y[1:n + 1] = y[0]
        if zero:
            y[1:n + 1] = 0.0
        return incident_features(y, pred_a, np.full(n + 2, temp), idx, 1, n)

    def test_after_hours(self):
        extra = 80.0 + np.array([3, -2, 4, -3, 2, -4, 1, 0.0])   # real loads are never exactly constant
        f = self._feat("2025-03-01 21:00", 8, extra)
        self.assertEqual(diagnose(f).cause, "hvac_after_hours")

    def test_meter_flatline_and_zero(self):
        self.assertEqual(diagnose(self._feat("2025-03-01 09:00", 10, 0.0, flat=True)).cause, "meter_fault")
        self.assertEqual(diagnose(self._feat("2025-03-01 09:00", 6, 0.0, zero=True)).cause, "meter_fault")

    def test_unexplained_when_weak(self):
        f = self._feat("2025-03-01 09:00", 5, np.array([1.0, -1, 2, -2, 1]))
        d = diagnose(f)
        self.assertEqual(d.cause, "unexplained")
        self.assertEqual(d.confidence, "low")

    def test_candidates_sorted(self):
        d = diagnose(self._feat("2025-03-01 21:00", 8, 80.0 + np.array([3, -2, 4, -3, 2, -4, 1, 0.0])))
        scores = [s for _, s in d.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestKnowledge(unittest.TestCase):
    def test_cause_grounding_first(self):
        kb = KnowledgeBase()
        hits = kb.retrieve("夜間 空調", cause="hvac_after_hours", k=2)
        self.assertEqual(hits[0].id, "hvac_after_hours")

    def test_every_entry_has_source_and_flag(self):
        for e in KnowledgeBase().entries:
            self.assertTrue(e["source"])
            self.assertIn("verified", e)
            if e["verified"]:
                self.assertTrue(e["url"])


class TestPlanner(unittest.TestCase):
    now = pd.Timestamp("2026-01-01 00:00")

    def test_explicit_range_and_building(self):
        p = parse_query("2025-07-10 到 2025-07-20 工學院 為何用電偏高", self.now, ["ENG", "LIB"])
        self.assertEqual(p.buildings, ["ENG"])
        self.assertEqual(p.start, pd.Timestamp("2025-07-10"))
        self.assertEqual(p.end, pd.Timestamp("2025-07-20 23:00"))
        self.assertEqual(p.intent, "diagnose")

    def test_default_all_buildings_and_assumption(self):
        p = parse_query("用電狀況", self.now, ["ENG", "LIB"])
        self.assertEqual(p.buildings, ["ENG", "LIB"])
        self.assertTrue(p.assumptions)

    def test_last_week(self):
        p = parse_query("library last week", pd.Timestamp("2025-06-18"), ["ENG", "LIB"])  # Wednesday
        self.assertEqual(p.buildings, ["LIB"])
        self.assertEqual(p.start, pd.Timestamp("2025-06-09"))


class TestQuality(unittest.TestCase):
    def test_flat_and_missing(self):
        y = pd.Series([1.0, 2, 3] + [5.0] * 10 + [4, 6], index=pd.date_range("2025-01-01", periods=15, freq="h"))
        q = quality_report(y)
        self.assertGreaterEqual(q["longest_flat_run"], 9)
        self.assertTrue(q["warnings"])
        y2 = y.copy(); y2.iloc[:3] = np.nan
        self.assertAlmostEqual(quality_report(y2)["missing_frac"], 3 / 15)


class TestSimulator(unittest.TestCase):
    def test_reproducible_and_labelled(self):
        a, b = generate_dataset(3), generate_dataset(3)
        pd.testing.assert_frame_equal(a.meter, b.meter)
        self.assertGreater(len(a.truth), 10)
        self.assertTrue((a.truth["start"] >= pd.Timestamp("2025-01-01")).all())
        self.assertFalse(a.mask.loc[:"2024-12-31"].any().any())   # training year is clean

    def test_events_do_not_overlap_anomalies(self):
        d = generate_dataset(1)
        for _, t in d.truth.iterrows():
            ev = d.events[d.events["building"] == t["building"]]
            self.assertFalse(((ev["start"] < t["end"]) & (ev["end"] > t["start"])).any())
