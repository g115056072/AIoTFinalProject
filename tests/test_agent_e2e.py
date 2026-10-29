import unittest
import _path  # noqa: F401
import pandas as pd

from energy_detective.agent.agent import EnergyDetectiveAgent
from energy_detective.agent.llm import AnthropicExplainer
from energy_detective.data.loaders import load_csv_dataset, save_dataset
from energy_detective.data.simulator import generate_dataset


class TestAgentEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_dataset(0)
        cls.agent = EnergyDetectiveAgent(cls.data)

    def test_diagnoses_injected_after_hours_event(self):
        t = self.data.truth[(self.data.truth["cause"] == "hvac_after_hours")].sort_values("start").iloc[0]
        name = self.data.buildings[t["building"]].name
        q = f"{t['start']:%Y-%m-%d} 到 {t['end']:%Y-%m-%d} {name} 為何用電偏高"
        r = self.agent.ask(q)
        self.assertTrue(any(f.kind == "anomaly" and f.diagnosis.cause == "hvac_after_hours"
                            and f.incident.start <= t["end"] and f.incident.end >= t["start"]
                            for f in r.findings))
        self.assertIn("診斷報告", r.report_md)
        self.assertTrue(r.trace)

    def test_scheduled_event_is_not_an_alarm(self):
        ev = self.data.events.sort_values("start").iloc[8]    # an event in the test year
        name = self.data.buildings[ev["building"]].name
        r = self.agent.ask(f"{ev['start']:%Y-%m-%d} {name} 為何用電偏高")
        self.assertFalse(any(f.kind == "anomaly" for f in r.findings if f.incident.start.date() == ev["start"].date()))

    def test_report_states_baseline_quality(self):
        r = self.agent.ask("2025-03-01 到 2025-03-07 工學院 用電")
        self.assertIn("ASHRAE Guideline 14", r.report_md)

    def test_llm_explainer_falls_back_offline(self):
        txt = AnthropicExplainer().explain({"kind": "expected_event", "event_name": "x"})
        self.assertIn("預期變化", txt)


class TestCsvRoundTrip(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile
        d = generate_dataset(2)
        with tempfile.TemporaryDirectory() as tmp:
            save_dataset(d, tmp)
            d2 = load_csv_dataset(tmp)
        self.assertEqual(d2.meter.shape, d.meter.shape)
        self.assertEqual(list(d2.meter.columns), list(d.meter.columns))
        self.assertAlmostEqual(float(d2.meter.iloc[100, 0]), float(d.meter.iloc[100, 0]), places=6)


if __name__ == "__main__":
    unittest.main()
