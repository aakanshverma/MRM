"""Alert register: schema, severity arithmetic and non-misleading behaviour."""

from __future__ import annotations

import pandas as pd
import pytest

from model_monitoring.alert_engine import (
    ALERT_COLUMNS,
    STATUS_DATA_GAP,
    STATUS_ESCALATED,
    STATUS_NONE,
    STATUS_OPEN,
    alert_counts,
    build_alerts,
    overall_status,
    priority_investigations,
    top_concerns,
)
from model_monitoring.config import Severity, severity_rank, worst_severity


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


def test_severity_ordering_puts_grey_between_green_and_amber():
    """GREY means 'unknown', which is worse than GREEN but not an escalation."""
    assert severity_rank("GREEN") < severity_rank("GREY") < severity_rank("AMBER") \
        < severity_rank("RED")
    assert worst_severity(["GREEN", "GREY", "AMBER"]) is Severity.AMBER
    assert worst_severity(["GREEN", "RED", "AMBER"]) is Severity.RED
    assert worst_severity([]) is Severity.GREEN
    assert severity_rank(None) == severity_rank("GREY")
    assert severity_rank("nonsense") == severity_rank("GREY")


# ---------------------------------------------------------------------------
# Register construction
# ---------------------------------------------------------------------------


def test_alerts_have_the_required_schema(pipeline_result):
    """Section 9 requires exactly these columns."""
    alerts = pipeline_result.report_inputs.alerts
    assert list(alerts.columns) == ALERT_COLUMNS
    for column in (
        "MODEL", "QUARTER", "METRIC", "VARIABLE_OR_SEGMENT", "METRIC_VALUE",
        "THRESHOLD", "SEVERITY", "REFERENCE_PERIOD", "MONITORING_PERIOD",
        "EXPLANATION", "RECOMMENDED_ACTION", "INVESTIGATION_STATUS",
    ):
        assert column in alerts.columns
    assert set(alerts["SEVERITY"]) <= {"GREEN", "AMBER", "RED", "GREY"}


def test_every_alert_explains_itself_and_proposes_an_action(pipeline_result):
    alerts = pipeline_result.report_inputs.alerts
    assert (alerts["EXPLANATION"].fillna("").str.len() > 10).all()
    assert (alerts["RECOMMENDED_ACTION"].fillna("").str.len() > 10).all()


def test_investigation_status_follows_severity(pipeline_result):
    alerts = pipeline_result.report_inputs.alerts
    mapping = {
        "RED": STATUS_ESCALATED, "AMBER": STATUS_OPEN,
        "GREY": STATUS_DATA_GAP, "GREEN": STATUS_NONE,
    }
    for severity, expected in mapping.items():
        subset = alerts[alerts["SEVERITY"] == severity]
        if not subset.empty:
            assert set(subset["INVESTIGATION_STATUS"]) == {expected}


def test_manual_review_fields_are_left_for_the_model_owner(pipeline_result):
    alerts = pipeline_result.report_inputs.alerts
    assert set(alerts["MANUAL_REVIEW_STATUS"]) == {"NOT STARTED"}
    assert set(alerts["MODEL_OWNER_NOTES"]) == {""}


def test_register_covers_every_metric_family(pipeline_result):
    alerts = pipeline_result.report_inputs.alerts
    categories = set(alerts["CATEGORY"])
    assert "POPULATION STABILITY" in categories
    assert "CHARACTERISTIC STABILITY" in categories
    assert "SEGMENT MIGRATION" in categories
    assert "REFERENCE ADEQUACY" in categories
    assert any(category.startswith("DATA QUALITY") for category in categories)


def test_reference_period_and_mode_appear_on_stability_alerts(pipeline_result):
    alerts = pipeline_result.report_inputs.alerts
    psi = alerts[alerts["CATEGORY"] == "POPULATION STABILITY"]
    assert not psi.empty
    assert (psi["REFERENCE_PERIOD"].fillna("").str.len() > 0).all()
    # each model's own window must appear, not a shared one
    fixed = psi[psi["METRIC"] == "PSI (fixed reference)"]
    windows = fixed.groupby("MODEL")["REFERENCE_PERIOD"].unique()
    assert all(len(values) == 1 for values in windows)
    assert len({tuple(values) for values in windows}) == len(windows)


def test_no_alert_recommends_automatic_model_rejection(pipeline_result):
    """A breach triggers investigation, never automatic rejection."""
    alerts = pipeline_result.report_inputs.alerts
    actions = " ".join(alerts["RECOMMENDED_ACTION"].fillna("").tolist()).lower()
    for phrase in ("retire the model", "reject the model", "decommission the scorecard"):
        assert phrase not in actions
    red = alerts[alerts["SEVERITY"] == "RED"]
    if not red.empty:
        assert any("investigation" in text.lower() or "escalate" in text.lower()
                   for text in red["RECOMMENDED_ACTION"])


def test_grey_alerts_never_claim_a_breach(pipeline_result):
    alerts = pipeline_result.report_inputs.alerts
    grey = alerts[alerts["SEVERITY"] == "GREY"]
    assert not grey.empty
    for text in grey["RECOMMENDED_ACTION"]:
        lowered = text.lower()
        assert any(
            phrase in lowered
            for phrase in (
                "insufficient", "cannot", "not measurable", "resolve the data",
                "re-assess", "do not infer", "larger", "confirm whether",
                "restore the characteristic", "read the segment",
            )
        ), text


# ---------------------------------------------------------------------------
# Counting and status
# ---------------------------------------------------------------------------


def test_alert_counts_always_include_every_severity():
    counts = alert_counts(pd.DataFrame(columns=ALERT_COLUMNS))
    assert counts == {"GREEN": 0, "AMBER": 0, "RED": 0, "GREY": 0}


def test_overall_status_escalates_on_red():
    def frame(severities):
        return pd.DataFrame({"SEVERITY": severities})

    assert overall_status(frame(["GREEN", "AMBER", "RED"])).startswith("RED")
    assert overall_status(frame(["GREEN", "AMBER"])).startswith("AMBER")
    assert overall_status(frame(["GREEN", "GREY"])) == "GREEN WITH DATA GAPS"
    assert overall_status(frame(["GREY"])).startswith("GREY")
    assert overall_status(frame(["GREEN"])).startswith("GREEN - WITHIN")


def test_counts_reconcile_with_the_register(pipeline_result):
    alerts = pipeline_result.report_inputs.alerts
    counts = alert_counts(alerts)
    assert sum(counts.values()) == len(alerts)
    assert counts == pipeline_result.alert_summary


# ---------------------------------------------------------------------------
# Dashboard extracts
# ---------------------------------------------------------------------------


def test_top_concerns_are_deduplicated_and_ranked(pipeline_result):
    alerts = pipeline_result.report_inputs.alerts
    concerns = top_concerns(alerts, n=5)
    assert len(concerns) <= 5
    assert list(concerns["RANK"]) == list(range(1, len(concerns) + 1))
    # one concern per model+metric: five rows about five columns of one model
    # is not five concerns
    assert not concerns.duplicated(subset=["MODEL", "METRIC"]).any()
    ranks = concerns["SEVERITY"].map(severity_rank).tolist()
    assert ranks == sorted(ranks, reverse=True)


def test_top_concerns_handles_a_clean_run():
    clean = pd.DataFrame(
        [{column: None for column in ALERT_COLUMNS} for _ in range(2)]
    )
    clean["SEVERITY"] = "GREEN"
    clean["METRIC_VALUE"] = 0.01
    concerns = top_concerns(clean, n=5)
    assert len(concerns) == 1
    assert concerns["SEVERITY"].iloc[0] == "GREEN"
    assert "No metric breached" in concerns["CONCERN"].iloc[0]


def test_priority_investigations_only_lists_actionable_items(pipeline_result):
    alerts = pipeline_result.report_inputs.alerts
    priorities = priority_investigations(alerts, n=10)
    assert set(priorities["SEVERITY"]) <= {"RED", "AMBER"}
    assert list(priorities["PRIORITY"]) == list(range(1, len(priorities) + 1))
    assert not priorities.duplicated(
        subset=["MODEL", "METRIC", "VARIABLE_OR_SEGMENT"]
    ).any()


def test_priority_investigations_on_a_clean_run():
    clean = pd.DataFrame([{column: None for column in ALERT_COLUMNS}])
    clean["SEVERITY"] = "GREEN"
    priorities = priority_investigations(clean)
    assert len(priorities) == 1
    assert priorities["INVESTIGATION_STATUS"].iloc[0] == STATUS_NONE


# ---------------------------------------------------------------------------
# Thresholds are honoured, not re-derived
# ---------------------------------------------------------------------------


def test_alert_severity_matches_the_source_metric_severity(pipeline_result):
    """The register must not re-classify: it carries the metric's own verdict."""
    inputs = pipeline_result.report_inputs
    alerts = inputs.alerts
    psi_alerts = alerts[alerts["METRIC"] == "PSI (fixed reference)"]
    psi_source = inputs.psi.summary[inputs.psi.summary["COMPARISON"] == "VS_REFERENCE"]
    merged = psi_alerts.merge(
        psi_source, left_on=["MODEL", "QUARTER"], right_on=["MODEL", "QUARTER"],
        suffixes=("_alert", "_source"),
    )
    assert not merged.empty
    assert (merged["SEVERITY_alert"] == merged["SEVERITY_source"]).all()


def test_overridden_thresholds_change_the_register(frame, config):
    from model_monitoring.main import run_pipeline

    config.thresholds.psi.amber = 0.0001
    config.thresholds.psi.red = 0.0002
    strict = run_pipeline(
        config, frame=frame, write_excel=False, output=lambda *a, **k: None
    )
    assert strict.alert_summary["RED"] > 0
    assert strict.status.startswith("RED")


def test_green_rows_can_be_excluded_from_the_register(frame, config):
    from model_monitoring.main import run_pipeline

    config.runtime.alert_include_green = False
    result = run_pipeline(
        config, frame=frame, write_excel=False, output=lambda *a, **k: None
    )
    csi_green = result.report_inputs.alerts[
        (result.report_inputs.alerts["CATEGORY"] == "CHARACTERISTIC STABILITY")
        & (result.report_inputs.alerts["SEVERITY"] == "GREEN")
    ]
    assert csi_green.empty


def test_priority_investigations_lead_with_breadth_not_one_models_columns(pipeline_result):
    """Eight consecutive rows about one model crowds out every other finding."""
    alerts = pipeline_result.report_inputs.alerts
    actionable = alerts[alerts["SEVERITY"].isin(["RED", "AMBER"])]
    families = actionable.groupby(["MODEL", "METRIC"]).ngroups
    priorities = priority_investigations(alerts, n=8)
    if families >= 3 and len(priorities) >= 3:
        # the first three rows must come from three different model+metric families
        head = priorities.head(3)
        assert len(set(zip(head["MODEL"], head["METRIC"]))) == 3
