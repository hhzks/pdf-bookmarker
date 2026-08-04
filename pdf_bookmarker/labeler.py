"""Serve the line-labeling heading detector.

Classifies every extracted line as "not a heading" or its nesting level, and
reads the outline straight off the labels. Two consequences matter downstream:
titles are the page's own text, so they are exact by construction, and each
entry already knows its physical page — the locator has nothing to do.

Measured on the 76-document evaluation set (`--ignore-section-numbers`):

    font heuristics   0.6205 title F1
    LLM (Qwen3.5-2B)  0.7642
    this              0.7671   precision 0.8588, level accuracy 0.8368
    this + the LLM    0.8104   see merge.merge_outlines

The model is a pair of fitted scikit-learn estimators produced by
`training/train_line_labeler.py --save-model`. scikit-learn and joblib are the
`[labeler]` extra and are imported lazily, so a pipeline with no labeler
configured needs neither.
"""
from dataclasses import dataclass
from pathlib import Path

from .extractor import Line
from .line_labeler import FEATURE_NAMES, entries_from_labels, feature_vector, line_features
from .models import OutlineEntry

_REQUIRED = ("detector", "leveler", "threshold", "feature_names")


class LabelerError(Exception):
    """The labeler model could not be loaded or is not usable."""


@dataclass
class Labeler:
    """A loaded heading detector. Construct via `Labeler.load`."""

    detector: object       # binary: is this line a heading
    leveler: object        # multiclass: how deep, fitted on headings only
    threshold: float       # tuned on validation, not left at 0.5
    feature_names: list[str]

    @classmethod
    def load(cls, path: Path | str) -> "Labeler":
        """Load a model bundle, failing loudly rather than scoring wrongly."""
        path = Path(path)
        if not path.is_file():
            raise LabelerError(f"labeler model not found: {path}")
        try:
            import joblib
        except ImportError as exc:
            raise LabelerError(
                'joblib is not installed; run pip install "pdf-bookmarker[labeler]"'
            ) from exc
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            raise LabelerError(f"cannot read labeler model {path}: {exc}") from exc
        if not isinstance(bundle, dict):
            raise LabelerError(f"{path} is not a labeler model bundle")
        missing = [key for key in _REQUIRED if key not in bundle]
        if missing:
            raise LabelerError(
                f"labeler model {path} is missing {', '.join(missing)}"
            )
        # A model fitted on a different feature set would still predict — on
        # the wrong columns, silently. Refuse instead.
        if list(bundle["feature_names"]) != list(FEATURE_NAMES):
            raise LabelerError(
                f"labeler model {path} was fitted on different features "
                f"({len(bundle['feature_names'])} columns, this build expects "
                f"{len(FEATURE_NAMES)}); retrain or install a matching model"
            )
        return cls(
            detector=bundle["detector"],
            leveler=bundle["leveler"],
            threshold=float(bundle["threshold"]),
            feature_names=list(bundle["feature_names"]),
        )

    @staticmethod
    def _matrix(rows):
        """float32, matching training exactly.

        Handing sklearn a list of Python floats gets it converted to float64.
        HistGradientBoosting bins its features, so a value that sat just inside
        a bin edge at training precision can land the other side of it at
        serving precision and flip the prediction. Measured on the evaluation
        set, that cost 1.3 title F1 (0.7540 against the harness's 0.7671) and
        showed up as the detector firing on contents rows.

        numpy ships with scikit-learn, so this adds no dependency.
        """
        import numpy

        return numpy.asarray(rows, dtype=numpy.float32)

    def detect(self, lines: list[Line], page_count: int) -> list[OutlineEntry]:
        """Return the outline this model reads off the document's lines."""
        rows = line_features(lines)
        if not rows:
            return []
        X = self._matrix([feature_vector(row, page_count) for row in rows])
        scores = self.detector.predict_proba(X)
        hits = [i for i, p in enumerate(scores) if p[1] >= self.threshold]
        if not hits:
            return []
        # Level is only defined for headings, and its classifier was fitted on
        # headings alone, so it runs over the detected lines only.
        levels = self.leveler.predict(self._matrix([X[i] for i in hits]))
        labels = [0] * len(rows)
        for i, level in zip(hits, levels):
            labels[i] = int(level)
        return [
            OutlineEntry(
                title=entry["title"],
                level=entry["level"],
                page=entry["page"],
                y=rows[entry["index"]]["y"],
            )
            for entry in entries_from_labels(rows, labels)
        ]
