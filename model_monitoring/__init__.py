"""End-to-end credit risk scorecard monitoring pipeline.

A configuration-driven framework that turns a customer-level extract of
scorecard outputs into a quarterly model monitoring pack:

* MER (Model Effectiveness Report) metrics per model and quarter
* PSI against each scorecard's own reference period, plus quarter-on-quarter PSI
* CSI for every monitored characteristic, subscore and final score
* Segment migration (6x6 matrices, upgrades / downgrades / net risk movement)
* Data quality, outcome performance and a unified alert register
* A formatted, chart-rich Excel workbook

See ``README.md`` for setup and ``docs/METRICS.md`` for every formula.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
