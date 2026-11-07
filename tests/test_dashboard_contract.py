"""Static contract between dashboard controls, help content and API choices."""
from html.parser import HTMLParser
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "static/index.html").read_text(encoding="utf-8")
JS = (ROOT / "static/app.js").read_text(encoding="utf-8")


class DashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.labels: set[str] = set()
        self.options: dict[str, set[str]] = {}
        self.active_select: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "label" and attributes.get("for"):
            self.labels.add(attributes["for"])
        if tag == "select":
            self.active_select = attributes.get("id")
            if self.active_select:
                self.options[self.active_select] = set()
        if tag == "option" and self.active_select and attributes.get("value"):
            self.options[self.active_select].add(attributes["value"])

    def handle_endtag(self, tag: str) -> None:
        if tag == "select":
            self.active_select = None


def test_help_targets_and_selectors_exist() -> None:
    page = DashboardParser()
    page.feed(HTML)
    help_block = JS.split("const helpContent = [", 1)[1].split("\n];", 1)[0]
    targets = re.findall(r"\['#([^']+)'", help_block)
    assert len(targets) == 25
    assert set(targets) <= page.ids
    assert {"method", "var-model", "stress-preset"} <= page.labels


def test_every_method_and_risk_model_has_explanation() -> None:
    page = DashboardParser()
    page.feed(HTML)
    method_text = JS.split("const allocationHelp = {", 1)[1].split("\n};", 1)[0]
    model_text = JS.split("const modelHelp = {", 1)[1].split("\n};", 1)[0]
    assert page.options["method"] == set(re.findall(r"^  (\w+):", method_text, re.M))
    assert page.options["var-model"] == set(re.findall(r"^  (\w+):", model_text, re.M))
    assert "none" in page.options["stress-preset"]
    assert "stress_preset: $('stress-preset').value" in JS


def test_confidence_headings_follow_selected_level() -> None:
    assert "`VaR ${confidenceLabel} / dia`" in JS
    assert "`Expected Shortfall ${confidenceLabel}`" in JS


def test_risk_free_mode_keeps_legacy_links() -> None:
    assert "params.has('risk-free') ? 'manual' : 'automatic'" in JS
    assert "'rf-mode'" in JS
    assert 'risk_free_annual: $(\'rf-mode\').value === \'manual\'' in JS
