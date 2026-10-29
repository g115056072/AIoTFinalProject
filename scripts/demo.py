"""Run the agent on a query: python scripts/demo.py --query "2025-07-10 到 2025-07-20 工學院 為何用電偏高"

With no --data it simulates a campus (seed 0). With --data DIR it loads CSVs
(see data/loaders.py for the schema), so real campus data can be used.
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from energy_detective.agent.agent import EnergyDetectiveAgent
from energy_detective.agent.llm import AnthropicExplainer, TemplateExplainer
from energy_detective.data.loaders import load_csv_dataset
from energy_detective.data.simulator import generate_dataset
from energy_detective.report.report import plot_result

ap = argparse.ArgumentParser()
ap.add_argument("--query", default=None)
ap.add_argument("--data", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--llm", action="store_true", help="use Anthropic API for wording (needs ANTHROPIC_API_KEY)")
ap.add_argument("--out", default="results/demo")
a = ap.parse_args()

data = load_csv_dataset(a.data) if a.data else generate_dataset(a.seed)
query = a.query
if not query:
    t = data.truth.sort_values("start").iloc[0]
    query = f"{t['start']:%Y-%m-%d} 到 {t['end']:%Y-%m-%d} {data.buildings[t['building']].name} 為何用電偏高"
agent = EnergyDetectiveAgent(data, explainer=AnthropicExplainer() if a.llm else TemplateExplainer())
res = agent.ask(query)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "report.md").write_text(res.report_md, encoding="utf-8")
plot_result(res, out / "report.png")
print(res.report_md)
print(f"\n已輸出：{out / 'report.md'}、{out / 'report.png'}")
