"""Static HTML/API agreement; no browser, network or rendering required."""

from html.parser import HTMLParser
from pathlib import Path
import unittest


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.labels, self.images, self.external_scripts = set(), [], [], []
        self.open_tags, self.ancestors, self.inputs = [], {}, {}

    def handle_starttag(self, tag, attrs):
        value = dict(attrs)
        if "id" in value:
            self.ids.add(value["id"])
            self.ancestors[value["id"]] = [attrs for _, attrs in self.open_tags]
        if tag == "label" and "for" in value:
            self.labels.append(value["for"])
        if tag == "img":
            self.images.append(value)
        if tag == "script" and "src" in value:
            self.external_scripts.append(value["src"])
        if tag == "input" and "id" in value:
            self.inputs[value["id"]] = value
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.open_tags.append((tag, value))

    def handle_endtag(self, tag):
        for index in range(len(self.open_tags)-1, -1, -1):
            if self.open_tags[index][0] == tag:
                del self.open_tags[index:]
                break


class UIContractTests(unittest.TestCase):
    def test_controls_credits_and_json_fields(self):
        text = (Path(__file__).resolve().parents[1]/"aws/static/index.html").read_text()
        page = Page()
        page.feed(text)
        self.assertFalse(page.external_scripts)
        self.assertTrue(all(label in page.ids for label in page.labels))
        self.assertTrue(all(image.get("alt") for image in page.images))
        for value in ("/api/samples", "/api/inspect", "/api/upload-url", "/api/review",
                      "sample_id", "budget_frac", "detector", "trace_url", "review_token",
                      "X-Inspection-Token", "X-Upload-Token", "calibration_cost_frac", "total_cost_frac",
                      "Poly Haven (CC0) — Powered by Poly Haven", "Özgenel (2019)",
                      "Built with OpenCV 5 and AWS", "prefers-reduced-motion"):
            self.assertIn(value, text)
        for ident in ("view", "run", "budget", "detector", "sample", "reviews", "report", "status"):
            self.assertIn(ident, page.ids)

    def test_evidence_states_and_snapshot_replay_are_explicit(self):
        text = (Path(__file__).resolve().parents[1]/"aws/static/index.html").read_text()
        for marker in ("coarse_candidates", "record_candidate", "clear_candidates",
                       "merged_reports", "item.scale===1", "Unconfirmed candidates",
                       "Confirmed model reports", "Legacy reports — resolution unverified",
                       "confirmed-findings", "candidate-findings"):
            self.assertIn(marker, text)
        self.assertIn("if(Array.isArray(step.merged_reports))reports=step.merged_reports", text)
        self.assertNotIn("if(s.detections)for", text)

    def test_gaze_is_limited_and_legend_stays_outside_the_image(self):
        text = (Path(__file__).resolve().parents[1]/"aws/static/index.html").read_text()
        page = Page()
        page.feed(text)
        self.assertIn("view-legend", page.ids)
        self.assertFalse(any("canvas-wrap" in attrs.get("class", "").split()
                             for attrs in page.ancestors["view-legend"]))
        self.assertNotIn("checked", page.inputs["heat"])
        self.assertIn("inspections.slice(-40)", text)
        self.assertIn("ctx.globalAlpha=.22", text)
        self.assertIn("ctx.globalAlpha=.28", text)
        self.assertIn("last 40 inspections", text)
        self.assertNotIn("Math.atan2", text)
        self.assertNotIn("ctx.strokeText", text)

    def test_coarse_reports_have_their_own_label_and_count(self):
        text = (Path(__file__).resolve().parents[1]/"aws/static/index.html").read_text()
        for marker in ("function isCoarseReport(item)", "item.status==='coarse_report'",
                       "item.report_mode==='coarse_only'", "coarseReports=reports.filter(isCoarseReport)",
                       "Coarse-only reports — low-resolution evidence",
                       "['Coarse-only reports (low resolution)',coarseReports.length]",
                       "item.scale!==1&&!isCoarseReport(item)"):
            self.assertIn(marker, text)
