"""Tests for pdf_bookmarker/labeler.py — serving the line-labeling model."""
import pytest

from pdf_bookmarker import labeler
from pdf_bookmarker.extractor import Line
from pdf_bookmarker.models import OutlineEntry


class FakeClassifier:
    """Stands in for a fitted sklearn estimator.

    `scores` is read positionally, one value per row, so a test can dictate
    exactly which lines the detector fires on.
    """

    def __init__(self, scores=None, level=1):
        self.scores = scores
        self.level = level
        self.seen = []

    def predict_proba(self, X):
        self.seen.append(X)
        scores = self.scores if self.scores is not None else [1.0] * len(X)
        return [[1.0 - s, s] for s in scores]

    def predict(self, X):
        self.seen.append(X)
        return [self.level] * len(X)


def make(scores=None, level=1, threshold=0.5):
    return labeler.Labeler(
        detector=FakeClassifier(scores),
        leveler=FakeClassifier(level=level),
        threshold=threshold,
        feature_names=list(labeler.FEATURE_NAMES),
    )


def lines():
    return [
        Line("1 Introduction", 0, 72, 72, 16, True),
        Line("body text here", 0, 72, 100, 10, False),
        Line("2 Methods", 3, 72, 72, 16, True),
    ]


def test_detects_the_lines_above_the_threshold():
    model = make(scores=[0.9, 0.1, 0.9])
    entries = model.detect(lines(), page_count=4)
    assert [e.title for e in entries] == ["1 Introduction", "2 Methods"]


def test_entries_carry_the_exact_physical_page():
    """The labeler knows the page outright; nothing needs locating."""
    model = make(scores=[0.9, 0.1, 0.9])
    entries = model.detect(lines(), page_count=4)
    assert [e.page for e in entries] == [0, 3]
    assert all(e.y is not None for e in entries)


def test_titles_are_the_line_text_verbatim():
    model = make(scores=[0.9, 0.1, 0.1])
    assert model.detect(lines(), page_count=4)[0].title == "1 Introduction"


def test_threshold_is_respected():
    model = make(scores=[0.6, 0.6, 0.6], threshold=0.8)
    assert model.detect(lines(), page_count=4) == []


def test_level_comes_from_the_level_classifier():
    model = make(scores=[0.9, 0.1, 0.1], level=3)
    assert model.detect(lines(), page_count=4)[0].level == 3


def test_no_lines_gives_no_entries():
    assert make().detect([], page_count=1) == []


def test_the_level_classifier_only_sees_detected_lines():
    """Running it over every line would waste most of the work."""
    model = make(scores=[0.9, 0.1, 0.1])
    model.detect(lines(), page_count=4)
    assert len(model.leveler.seen[0]) == 1


def test_the_level_classifier_is_skipped_when_nothing_is_detected():
    model = make(scores=[0.1, 0.1, 0.1])
    model.detect(lines(), page_count=4)
    assert model.leveler.seen == []


# --- loading -----------------------------------------------------------------

def test_loading_a_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(labeler.LabelerError, match="not found"):
        labeler.Labeler.load(tmp_path / "nope.joblib")


def test_loading_a_non_model_file_is_a_clear_error(tmp_path):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"not a model")
    with pytest.raises(labeler.LabelerError):
        labeler.Labeler.load(path)


def test_a_bundle_missing_its_pieces_is_rejected(tmp_path):
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "partial.joblib"
    joblib.dump({"detector": FakeClassifier()}, path)
    with pytest.raises(labeler.LabelerError, match="missing"):
        labeler.Labeler.load(path)


def test_a_bundle_whose_features_disagree_is_rejected(tmp_path):
    """A model fitted on different features would score silently wrong."""
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "stale.joblib"
    joblib.dump(
        {
            "detector": FakeClassifier(),
            "leveler": FakeClassifier(),
            "threshold": 0.5,
            "feature_names": ["size_ratio", "bold"],  # an older, shorter set
        },
        path,
    )
    with pytest.raises(labeler.LabelerError, match="feature"):
        labeler.Labeler.load(path)


def test_a_well_formed_bundle_round_trips(tmp_path):
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "good.joblib"
    joblib.dump(
        {
            "detector": FakeClassifier(scores=[0.9, 0.1, 0.1]),
            "leveler": FakeClassifier(level=2),
            "threshold": 0.5,
            "feature_names": list(labeler.FEATURE_NAMES),
        },
        path,
    )
    model = labeler.Labeler.load(path)
    entries = model.detect(lines(), page_count=4)
    assert [(e.title, e.level) for e in entries] == [("1 Introduction", 2)]


# --- pipeline integration ----------------------------------------------------

def test_pipeline_is_unchanged_when_no_labeler_is_configured(toc_pdf, monkeypatch):
    from pdf_bookmarker import pipeline

    monkeypatch.delenv("PDF_BOOKMARKER_LABELER", raising=False)
    result = pipeline.process_pdf(toc_pdf, None, llm_mode="never")
    assert result.used_labeler is False
    assert result.used_toc is True          # still the TOC path
    assert result.entries


def test_result_fields_are_not_transposed(toc_pdf, monkeypatch):
    """warnings and used_labeler sit adjacent; a positional slip swaps them."""
    from pdf_bookmarker import pipeline

    monkeypatch.delenv("PDF_BOOKMARKER_LABELER", raising=False)
    result = pipeline.process_pdf(toc_pdf, None, llm_mode="never")
    assert isinstance(result.warnings, list)
    assert isinstance(result.used_labeler, bool)


def test_the_labeler_replaces_the_heuristic_path(toc_pdf, monkeypatch):
    from pdf_bookmarker import pipeline

    stub = make(scores=None)  # fires on every line
    monkeypatch.setattr(pipeline, "resolve_labeler", lambda path: stub)
    result = pipeline.process_pdf(toc_pdf, None, llm_mode="never")
    assert result.used_labeler is True
    assert result.used_toc is False


def test_the_env_var_configures_the_labeler(tmp_path, monkeypatch):
    from pdf_bookmarker import pipeline

    monkeypatch.setenv("PDF_BOOKMARKER_LABELER", str(tmp_path / "missing.joblib"))
    with pytest.raises(labeler.LabelerError):
        pipeline.resolve_labeler(None)


def test_an_explicit_path_beats_the_env_var(tmp_path, monkeypatch):
    from pdf_bookmarker import pipeline

    monkeypatch.setenv("PDF_BOOKMARKER_LABELER", str(tmp_path / "env.joblib"))
    with pytest.raises(labeler.LabelerError, match="explicit"):
        pipeline.resolve_labeler(tmp_path / "explicit.joblib")


def test_no_labeler_configured_resolves_to_none(monkeypatch):
    from pdf_bookmarker import pipeline

    monkeypatch.delenv("PDF_BOOKMARKER_LABELER", raising=False)
    assert pipeline.resolve_labeler(None) is None


def test_llm_entries_are_merged_into_the_labeler_outline(toc_pdf, monkeypatch):
    """With a labeler the LLM adds to the outline; without one it replaces it."""
    from pdf_bookmarker import llm as llm_module
    from pdf_bookmarker import pipeline
    from pdf_bookmarker.models import OutlineEntry

    class Backend:
        def parse_outline(self, context):
            return [OutlineEntry("Invented Section", 1, printed_page=1)]

    stub = make(scores=None)
    monkeypatch.setattr(pipeline, "resolve_labeler", lambda path: stub)
    monkeypatch.setattr(llm_module, "get_backend", lambda *a, **k: Backend())
    result = pipeline.process_pdf(toc_pdf, None, llm_mode="always")
    titles = [e.title for e in result.entries]
    assert "Invented Section" in titles
    assert len(titles) > 1          # the labeler's entries survived the merge


def test_a_labeler_that_finds_nothing_falls_back_to_the_heuristics(toc_pdf, monkeypatch):
    """Returning no outline would be worse than the path we already had."""
    from pdf_bookmarker import pipeline

    silent = make(scores=[0.0] * 500, threshold=0.5)
    monkeypatch.setattr(pipeline, "resolve_labeler", lambda path: silent)
    result = pipeline.process_pdf(toc_pdf, None, llm_mode="never")
    assert result.entries              # the heuristic outline, not an error
    assert result.used_labeler is False
    assert result.used_toc is True


# --- routing: when is the LLM worth calling on top of the labeler? -----------

class FirstN:
    """Fires on the first n lines, whatever the document — a fixed density."""

    def __init__(self, n):
        self.n = n

    def predict_proba(self, X):
        return [[0.0, 1.0] if i < self.n else [1.0, 0.0] for i in range(len(X))]


def firing_on(n):
    return labeler.Labeler(
        detector=FirstN(n),
        leveler=FakeClassifier(level=1),
        threshold=0.5,
        feature_names=list(labeler.FEATURE_NAMES),
    )


class CountingBackend:
    """Records every call and proposes one heading the labeler cannot know."""

    def __init__(self):
        self.calls = 0

    def parse_outline(self, context):
        self.calls += 1
        return [OutlineEntry("Invented Section", 1, printed_page=1)]


def _run_auto(pdf, monkeypatch, model, **kwargs):
    from pdf_bookmarker import llm as llm_module
    from pdf_bookmarker import pipeline

    backend = CountingBackend()
    monkeypatch.setattr(pipeline, "resolve_labeler", lambda path: model)
    monkeypatch.setattr(llm_module, "get_backend", lambda *a, **k: backend)
    result = pipeline.process_pdf(
        pdf, None, llm_mode="auto", api_key="test-key", **kwargs
    )
    return result, backend


def test_a_dense_labeler_outline_does_not_call_the_llm(toc_pdf, monkeypatch):
    """auto mode's whole point: skip the call when the cheap pass looks fine."""
    result, backend = _run_auto(toc_pdf, monkeypatch, make(scores=None))
    assert backend.calls == 0
    assert result.used_llm is False
    assert result.used_labeler is True


def test_a_sparse_labeler_outline_escalates_to_the_llm(toc_pdf, monkeypatch):
    """One heading across 5 pages is where the LLM adds the most."""
    result, backend = _run_auto(toc_pdf, monkeypatch, firing_on(1))
    assert backend.calls == 1
    assert result.used_llm is True
    assert "Invented Section" in [e.title for e in result.entries]


def test_the_escalated_call_merges_rather_than_replaces(toc_pdf, monkeypatch):
    """Routing must not throw away the labeler entries that triggered it."""
    result, _ = _run_auto(toc_pdf, monkeypatch, firing_on(1))
    assert len(result.entries) > 1


def test_the_density_threshold_is_configurable(toc_pdf, monkeypatch):
    """0 disables routing: the same sparse outline no longer pays for a call."""
    result, backend = _run_auto(toc_pdf, monkeypatch, firing_on(1), llm_density=0.0)
    assert backend.calls == 0
    assert result.used_llm is False


def test_a_raised_threshold_routes_a_denser_outline(toc_pdf, monkeypatch):
    result, backend = _run_auto(toc_pdf, monkeypatch, make(scores=None), llm_density=99.0)
    assert backend.calls == 1


def test_the_labeler_path_ignores_the_heuristic_confidence_rule(monkeypatch):
    """It fires on the documents that need the LLM least — measured, not assumed."""
    from pdf_bookmarker import llm as llm_module
    from pdf_bookmarker import pipeline

    calls = []
    monkeypatch.setattr(
        llm_module, "is_low_confidence", lambda *a: calls.append(a) or True
    )
    entries = [OutlineEntry(f"H{i}", 1, page=i) for i in range(20)]
    run, _ = pipeline.decide_llm(
        "auto", "key", entries, 0, False, 5, "anthropic:x", used_labeler=True
    )
    assert calls == []      # never consulted
    assert run is False     # 4 entries/page is dense


def test_the_heuristic_path_still_uses_the_confidence_rule(monkeypatch):
    """Density was measured on labeler outlines only; the other path is unchanged."""
    from pdf_bookmarker import llm as llm_module
    from pdf_bookmarker import pipeline

    calls = []
    monkeypatch.setattr(
        llm_module, "is_low_confidence", lambda *a: calls.append(a) or True
    )
    entries = [OutlineEntry(f"H{i}", 1, page=i) for i in range(20)]
    run, _ = pipeline.decide_llm(
        "auto", "key", entries, 0, True, 5, "anthropic:x", used_labeler=False
    )
    assert len(calls) == 1
    assert run is True


def test_always_mode_still_calls_the_llm_on_a_dense_outline(toc_pdf, monkeypatch):
    """--llm is an instruction, not a hint; routing must not override it."""
    from pdf_bookmarker import llm as llm_module
    from pdf_bookmarker import pipeline

    backend = CountingBackend()
    monkeypatch.setattr(pipeline, "resolve_labeler", lambda path: make(scores=None))
    monkeypatch.setattr(llm_module, "get_backend", lambda *a, **k: backend)
    result = pipeline.process_pdf(toc_pdf, None, llm_mode="always")
    assert backend.calls == 1
    assert result.used_llm is True


def test_features_reach_the_model_as_float32():
    """float64 shifts values across HistGradientBoosting's bin edges.

    Training fits on float32; serving at float64 flipped enough borderline
    predictions to cost 1.3 title F1 before this was pinned.
    """
    numpy = pytest.importorskip("numpy")
    captured = {}

    class Recorder(FakeClassifier):
        def predict_proba(self, X):
            captured["dtype"] = X.dtype
            return super().predict_proba(X)

    model = labeler.Labeler(
        detector=Recorder(scores=[0.1, 0.1, 0.1]),
        leveler=FakeClassifier(),
        threshold=0.5,
        feature_names=list(labeler.FEATURE_NAMES),
    )
    model.detect(lines(), page_count=4)
    assert captured["dtype"] == numpy.float32


# --- loading once, not per document -------------------------------------------

@pytest.fixture
def clean_cache(monkeypatch):
    from pdf_bookmarker import pipeline

    monkeypatch.setattr(pipeline, "_CACHED_LABELER", None, raising=False)


def _count_loads(monkeypatch):
    from pdf_bookmarker import pipeline

    loaded = []
    monkeypatch.setattr(
        pipeline.labeler_module.Labeler, "load",
        classmethod(lambda cls, path: loaded.append(str(path)) or "the model"),
    )
    return loaded


def test_the_model_is_loaded_once_across_calls(tmp_path, monkeypatch, clean_cache):
    """A web worker resolves per job; unpickling the bundle each time is waste."""
    from pdf_bookmarker import pipeline

    path = tmp_path / "m.joblib"
    path.write_bytes(b"x")
    loaded = _count_loads(monkeypatch)
    assert pipeline.resolve_labeler(path) == "the model"
    assert pipeline.resolve_labeler(path) == "the model"
    assert len(loaded) == 1


def test_a_changed_model_file_is_reloaded(tmp_path, monkeypatch, clean_cache):
    """Swapping the model in place must not need a restart to take effect."""
    from pdf_bookmarker import pipeline

    path = tmp_path / "m.joblib"
    path.write_bytes(b"x")
    loaded = _count_loads(monkeypatch)
    pipeline.resolve_labeler(path)
    path.write_bytes(b"a longer bundle")
    pipeline.resolve_labeler(path)
    assert len(loaded) == 2


def test_a_second_path_is_loaded_separately(tmp_path, monkeypatch, clean_cache):
    from pdf_bookmarker import pipeline

    first, second = tmp_path / "a.joblib", tmp_path / "b.joblib"
    first.write_bytes(b"x")
    second.write_bytes(b"y")
    loaded = _count_loads(monkeypatch)
    pipeline.resolve_labeler(first)
    pipeline.resolve_labeler(second)
    pipeline.resolve_labeler(first)
    assert len(loaded) == 3


def test_a_missing_model_still_raises(tmp_path, monkeypatch, clean_cache):
    """Caching must not swallow a path that stopped existing."""
    from pdf_bookmarker import pipeline

    with pytest.raises(labeler.LabelerError, match="not found"):
        pipeline.resolve_labeler(tmp_path / "gone.joblib")
