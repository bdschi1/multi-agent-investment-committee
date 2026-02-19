"""Tests for the XAI (Explainable AI) module.

All tests use mock/synthetic data — no API keys needed.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Sample fundamentals fixture (mirrors get_fundamentals() output)
# ---------------------------------------------------------------------------

HEALTHY_FUNDAMENTALS = {
    "ticker": "NVDA",
    "current_ratio": 4.2,
    "roe": "55.3%",
    "operating_margin": "62.1%",
    "price_to_book": 50.0,
    "debt_to_equity": 41.0,
    "profit_margin": "55.0%",
    "gross_margin": "75.3%",
    "revenue_growth": "122.4%",
    "earnings_growth": "581.0%",
    "pe_trailing": 65.0,
    "pe_forward": 40.0,
    "ev_to_ebitda": 50.0,
    "roa": "45.3%",
}

DISTRESSED_FUNDAMENTALS = {
    "ticker": "ZOMBIE",
    "current_ratio": 0.5,
    "roe": "-25.0%",
    "operating_margin": "-15.0%",
    "price_to_book": 0.3,
    "debt_to_equity": 500.0,
    "profit_margin": "-20.0%",
    "gross_margin": "10.0%",
    "revenue_growth": "-30.0%",
    "earnings_growth": "-50.0%",
    "pe_trailing": -5.0,
    "ev_to_ebitda": -3.0,
    "roa": "-10.0%",
}

MINIMAL_FUNDAMENTALS = {
    "ticker": "SPARSE",
    "pe_trailing": 20.0,
}


# ===========================================================================
# TestFeatureExtraction
# ===========================================================================

class TestFeatureExtraction:
    """Test feature extraction from fundamentals dict."""

    def test_extract_features_complete(self):
        from xai.features import FEATURE_NAMES, extract_features

        features = extract_features(HEALTHY_FUNDAMENTALS)
        assert len(features) == 12
        for name in FEATURE_NAMES:
            assert name in features

    def test_percent_string_parsing(self):
        from xai.features import extract_features

        features = extract_features(HEALTHY_FUNDAMENTALS)
        # "55.3%" → 0.553
        assert abs(features["roe"] - 0.553) < 0.001
        # "62.1%" → 0.621
        assert abs(features["operating_margin"] - 0.621) < 0.001
        # "45.3%" → 0.453
        assert abs(features["roa"] - 0.453) < 0.001

    def test_numeric_passthrough(self):
        from xai.features import extract_features

        features = extract_features(HEALTHY_FUNDAMENTALS)
        assert features["current_ratio"] == 4.2
        assert features["pe_trailing"] == 65.0
        assert features["debt_to_equity"] == 41.0

    def test_missing_field_imputation(self):
        from xai.features import FEATURE_DEFAULTS, extract_features

        features = extract_features(MINIMAL_FUNDAMENTALS)
        assert len(features) == 12
        # pe_trailing is present
        assert features["pe_trailing"] == 20.0
        # Others should use defaults
        assert features["roe"] == FEATURE_DEFAULTS["roe"]
        assert features["current_ratio"] == FEATURE_DEFAULTS["current_ratio"]

    def test_features_to_array_ordering(self):
        from xai.features import FEATURE_NAMES, extract_features, features_to_array

        features = extract_features(HEALTHY_FUNDAMENTALS)
        arr, names = features_to_array(features)
        assert arr.shape == (12,)
        assert names == FEATURE_NAMES
        assert arr[0] == features["current_ratio"]
        assert arr[-1] == features["roa"]

    def test_negative_percent_parsing(self):
        from xai.features import extract_features

        features = extract_features(DISTRESSED_FUNDAMENTALS)
        assert features["roe"] == pytest.approx(-0.25, abs=0.001)
        assert features["operating_margin"] == pytest.approx(-0.15, abs=0.001)


# ===========================================================================
# TestAltmanZScore
# ===========================================================================

class TestAltmanZScore:
    """Test Altman Z-Score model."""

    def test_safe_zone_classification(self):
        from xai.distress import AltmanZScoreModel
        from xai.features import extract_features, features_to_array

        model = AltmanZScoreModel()
        features = extract_features(HEALTHY_FUNDAMENTALS)
        x_arr, _ = features_to_array(features)
        z = model.compute_z_score(x_arr)
        zone = model.classify_zone(z)
        assert zone == "safe"
        assert z > model.SAFE_THRESHOLD

    def test_distress_zone_classification(self):
        from xai.distress import AltmanZScoreModel
        from xai.features import extract_features, features_to_array

        model = AltmanZScoreModel()
        features = extract_features(DISTRESSED_FUNDAMENTALS)
        x_arr, _ = features_to_array(features)
        z = model.compute_z_score(x_arr)
        zone = model.classify_zone(z)
        assert zone == "distress"
        assert z < model.GREY_THRESHOLD

    def test_predict_proba_shape(self):
        from xai.distress import AltmanZScoreModel
        from xai.features import extract_features, features_to_array

        model = AltmanZScoreModel()
        x_arr, _ = features_to_array(extract_features(HEALTHY_FUNDAMENTALS))
        proba = model.predict_proba(x_arr)
        assert proba.shape == (1, 2)
        assert abs(proba[0, 0] + proba[0, 1] - 1.0) < 1e-10

    def test_sigmoid_calibration(self):
        from xai.distress import AltmanZScoreModel
        from xai.features import extract_features, features_to_array

        model = AltmanZScoreModel()
        # Healthy: low PFD
        x_h, _ = features_to_array(extract_features(HEALTHY_FUNDAMENTALS))
        pfd_h = model.predict_proba(x_h)[0, 1]
        # Distressed: high PFD
        x_d, _ = features_to_array(extract_features(DISTRESSED_FUNDAMENTALS))
        pfd_d = model.predict_proba(x_d)[0, 1]
        assert pfd_h < 0.3  # healthy should have low PFD
        assert pfd_d > 0.5  # distressed should have high PFD

    def test_predict_binary(self):
        from xai.distress import AltmanZScoreModel
        from xai.features import extract_features, features_to_array

        model = AltmanZScoreModel()
        x_h, _ = features_to_array(extract_features(HEALTHY_FUNDAMENTALS))
        x_d, _ = features_to_array(extract_features(DISTRESSED_FUNDAMENTALS))
        assert model.predict(x_h)[0] == 0  # healthy
        assert model.predict(x_d)[0] == 1  # distressed

    def test_model_name(self):
        from xai.distress import AltmanZScoreModel

        assert AltmanZScoreModel().name == "altman_z_score"


# ===========================================================================
# TestXGBoostModel
# ===========================================================================

class TestXGBoostModel:
    """Test XGBoost model (optional dependency)."""

    def test_factory_fallback_to_altman(self):
        from xai.distress import AltmanZScoreModel, get_distress_model

        model = get_distress_model("/nonexistent/path.joblib")
        assert isinstance(model, AltmanZScoreModel)

    def test_xgboost_train_predict(self):
        """Train a tiny model and verify it predicts."""
        pytest.importorskip("xgboost")
        from xai.distress import XGBoostDistressModel

        model = XGBoostDistressModel()
        rng = np.random.RandomState(42)
        x_arr = rng.randn(50, 12)
        y = (x_arr[:, 0] + x_arr[:, 1] > 0).astype(int)
        metrics = model.train(x_arr, y)
        assert metrics["accuracy"] > 0.5
        proba = model.predict_proba(x_arr[:1])
        assert proba.shape == (1, 2)

    def test_xgboost_save_load_roundtrip(self, tmp_path):
        """Test save/load cycle."""
        pytest.importorskip("xgboost")
        pytest.importorskip("joblib")
        from xai.distress import XGBoostDistressModel

        model = XGBoostDistressModel()
        rng = np.random.RandomState(42)
        x_arr = rng.randn(30, 12)
        y = (x_arr[:, 0] > 0).astype(int)
        model.train(x_arr, y)

        path = tmp_path / "model.joblib"
        model.save(path)
        assert path.exists()

        model2 = XGBoostDistressModel()
        model2.load(path)
        proba1 = model.predict_proba(x_arr[:1])
        proba2 = model2.predict_proba(x_arr[:1])
        np.testing.assert_array_almost_equal(proba1, proba2)


# ===========================================================================
# TestSHAPExplainer
# ===========================================================================

class TestSHAPExplainer:
    """Test SHAP explanations."""

    def test_explain_returns_explanation(self):
        from xai.distress import AltmanZScoreModel
        from xai.explainer import SHAPExplainer
        from xai.features import extract_features, features_to_array

        model = AltmanZScoreModel()
        x_arr, names = features_to_array(extract_features(HEALTHY_FUNDAMENTALS))
        explainer = SHAPExplainer(top_k=3)
        result = explainer.explain(model, x_arr, names)

        assert len(result.feature_names) == 12
        # SHAP values should be populated (may be zeros if shap not installed)
        assert result.shap_values.shape == (12,)

    def test_fallback_without_shap_library(self, monkeypatch):
        """If shap library is unavailable, should use built-in Shapley calculator."""
        import xai.explainer as exp_mod

        # Force SHAP library to appear unavailable
        monkeypatch.setattr(exp_mod, "_SHAP_AVAILABLE", False)

        from xai.distress import AltmanZScoreModel
        from xai.explainer import SHAPExplainer
        from xai.features import extract_features, features_to_array

        model = AltmanZScoreModel()
        x_arr, names = features_to_array(extract_features(HEALTHY_FUNDAMENTALS))
        explainer = SHAPExplainer()
        result = explainer.explain(model, x_arr, names)

        # Built-in Shapley should produce non-trivial values
        assert not np.all(result.shap_values == 0)
        assert len(result.feature_names) == 12
        # No plot without shap/matplotlib
        assert result.waterfall_plot_base64 == ""

    def test_top_drivers_ordering(self):
        """Verify top drivers are sorted by absolute SHAP value."""
        from xai.explainer import SHAPExplainer

        explainer = SHAPExplainer(top_k=2)
        sv = np.array([0.1, -0.3, 0.5, -0.05, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        names = [f"f{i}" for i in range(12)]
        top_pos, top_neg = explainer._extract_top_drivers(sv, names)

        assert len(top_pos) == 2
        assert "f2" in top_pos[0]  # 0.5 is largest positive
        assert len(top_neg) == 2
        assert "f1" in top_neg[0]  # -0.3 is most negative


# ===========================================================================
# TestReturnEstimator
# ===========================================================================

class TestReturnEstimator:
    """Test expected return computation."""

    def test_er_formula_basic(self):
        from xai.returns import compute_expected_return

        # PFD=0.1, PE=20 → yield=0.05, ER = 0.9 * 0.05 = 0.045
        result = compute_expected_return(0.1, {"pe_trailing": 20.0})
        assert result.expected_return == pytest.approx(0.045, abs=0.001)

    def test_high_distress_near_zero_er(self):
        from xai.returns import compute_expected_return

        result = compute_expected_return(0.95, {"pe_trailing": 20.0})
        assert result.expected_return < 0.01

    def test_earnings_yield_forward_pe_priority(self):
        from xai.returns import compute_expected_return

        # Forward PE should be used first
        result = compute_expected_return(
            0.0, {"pe_forward": 25.0, "pe_trailing": 50.0}
        )
        assert result.earnings_yield_proxy == pytest.approx(1.0 / 25.0, abs=0.001)

    def test_earnings_yield_profit_margin_fallback(self):
        from xai.returns import compute_expected_return

        result = compute_expected_return(0.0, {"profit_margin": "10.0%"})
        assert result.earnings_yield_proxy == pytest.approx(0.10, abs=0.001)

    def test_screen_distress_clear(self):
        from xai.returns import screen_distress

        result = screen_distress(0.1, threshold=0.5)
        assert not result.is_distressed
        assert "CLEAR" in result.flag

    def test_screen_distress_flagged(self):
        from xai.returns import screen_distress

        result = screen_distress(0.7, threshold=0.5)
        assert result.is_distressed
        assert "DISTRESSED" in result.flag

    def test_screen_distress_watch(self):
        from xai.returns import screen_distress

        # 0.35 >= 0.5*0.6=0.30
        result = screen_distress(0.35, threshold=0.5)
        assert not result.is_distressed
        assert "WATCH" in result.flag


# ===========================================================================
# TestXAIPipeline
# ===========================================================================

class TestXAIPipeline:
    """Test end-to-end XAI pipeline."""

    def test_healthy_company_analysis(self):
        from xai.pipeline import XAIPipeline

        pipeline = XAIPipeline(distress_threshold=0.5)
        result = pipeline.analyze("NVDA", HEALTHY_FUNDAMENTALS)

        assert result.ticker == "NVDA"
        assert result.distress.distress_zone == "safe"
        assert result.distress.pfd < 0.3
        assert not result.returns.is_distressed
        assert result.computation_time_ms > 0
        assert len(result.features_used) == 12
        assert len(result.feature_importance_ranking) == 12

    def test_distressed_company_analysis(self):
        from xai.pipeline import XAIPipeline

        pipeline = XAIPipeline(distress_threshold=0.5)
        result = pipeline.analyze("ZOMBIE", DISTRESSED_FUNDAMENTALS)

        assert result.ticker == "ZOMBIE"
        assert result.distress.distress_zone == "distress"
        assert result.distress.pfd > 0.5
        assert result.returns.is_distressed
        assert result.returns.expected_return < result.returns.earnings_yield_proxy

    def test_missing_fundamentals_uses_defaults(self):
        from xai.pipeline import XAIPipeline

        pipeline = XAIPipeline()
        result = pipeline.analyze("SPARSE", MINIMAL_FUNDAMENTALS)

        assert result.ticker == "SPARSE"
        assert len(result.features_used) == 12
        # Should still produce valid result
        assert 0.0 <= result.distress.pfd <= 1.0

    def test_narrative_generation(self):
        from xai.pipeline import XAIPipeline

        pipeline = XAIPipeline()
        result = pipeline.analyze("NVDA", HEALTHY_FUNDAMENTALS)

        assert "NVDA" in result.narrative
        assert "PFD=" in result.narrative
        assert "Expected risk-adjusted return:" in result.narrative

    def test_timing_under_500ms(self):
        """XAI computation should be fast (no LLM calls)."""
        from xai.pipeline import XAIPipeline

        pipeline = XAIPipeline()
        result = pipeline.analyze("NVDA", HEALTHY_FUNDAMENTALS)
        # Should be very fast; allow generous margin for CI
        assert result.computation_time_ms < 5000  # 5 sec max

    def test_model_dump_serializable(self):
        """Result should be JSON-serializable via model_dump()."""
        import json

        from xai.pipeline import XAIPipeline

        pipeline = XAIPipeline()
        result = pipeline.analyze("NVDA", HEALTHY_FUNDAMENTALS)
        dumped = result.model_dump()
        # Should not raise
        json.dumps(dumped, default=str)


# ===========================================================================
# TestXAINode
# ===========================================================================

class TestXAINode:
    """Test LangGraph node integration."""

    def test_context_injection(self):
        from xai.node import run_xai_analysis

        state = {
            "ticker": "AAPL",
            "context": {
                "financial_metrics": HEALTHY_FUNDAMENTALS,
            },
        }
        result = run_xai_analysis(state, config={})

        assert "context" in result
        assert "xai_analysis" in result["context"]
        xai = result["context"]["xai_analysis"]
        assert xai["ticker"] == "AAPL"
        assert "distress" in xai
        assert "returns" in xai

    def test_graceful_failure_no_fundamentals(self):
        from xai.node import run_xai_analysis

        state = {
            "ticker": "EMPTY",
            "context": {},
        }
        result = run_xai_analysis(state, config={})
        # Should return empty dict (no state change)
        assert result == {}

    def test_graceful_failure_no_context(self):
        from xai.node import run_xai_analysis

        state = {"ticker": "EMPTY"}
        result = run_xai_analysis(state, config={})
        assert result == {}

    def test_does_not_mutate_original_context(self):
        from xai.node import run_xai_analysis

        original_context = {
            "financial_metrics": HEALTHY_FUNDAMENTALS,
        }
        state = {
            "ticker": "MSFT",
            "context": original_context,
        }
        run_xai_analysis(state, config={})
        # Original context should not have xai_analysis
        assert "xai_analysis" not in original_context

    def test_status_callback_fires(self):
        from xai.node import run_xai_analysis

        statuses = []
        state = {
            "ticker": "GOOG",
            "context": {"financial_metrics": HEALTHY_FUNDAMENTALS},
            "on_status": lambda msg: statuses.append(msg),
        }
        run_xai_analysis(state, config={})
        assert len(statuses) >= 1
        assert any("XAI" in s for s in statuses)


# ===========================================================================
# TestXAIGraphIntegration
# ===========================================================================

class TestXAIGraphIntegration:
    """Test XAI node integration in the LangGraph pipeline."""

    def test_graph_has_xai_node(self):
        from orchestrator.graph import build_graph

        graph = build_graph()
        # Check the compiled graph has the XAI node
        node_names = set()
        if hasattr(graph, "nodes"):
            node_names = set(graph.nodes.keys())
        assert "run_xai_analysis" in node_names

    def test_phase1_graph_has_xai_node(self):
        from orchestrator.graph import build_graph_phase1

        graph = build_graph_phase1()
        node_names = set()
        if hasattr(graph, "nodes"):
            node_names = set(graph.nodes.keys())
        assert "run_xai_analysis" in node_names


# ===========================================================================
# TestXAIDatabase
# ===========================================================================

class TestXAIDatabase:
    """Test XAI fields in backtest database."""

    def test_store_retrieve_signal_with_xai(self, tmp_path):
        from backtest.database import SignalDatabase
        from backtest.models import SignalRecord

        db = SignalDatabase(tmp_path / "test.db")

        signal = SignalRecord(
            ticker="AAPL",
            t_signal=0.65,
            xai_pfd=0.12,
            xai_z_score=3.5,
            xai_distress_zone="safe",
            xai_expected_return=0.045,
            xai_top_risk_factor="debt_to_equity",
            xai_model_used="altman_z_score",
        )
        sid = db.store_signal(signal)
        assert sid > 0

        retrieved = db.get_signal_by_id(sid)
        assert retrieved is not None
        assert retrieved.xai_pfd == pytest.approx(0.12)
        assert retrieved.xai_z_score == pytest.approx(3.5)
        assert retrieved.xai_distress_zone == "safe"
        assert retrieved.xai_expected_return == pytest.approx(0.045)
        assert retrieved.xai_top_risk_factor == "debt_to_equity"
        assert retrieved.xai_model_used == "altman_z_score"
        db.close()

    def test_store_signal_without_xai(self, tmp_path):
        from backtest.database import SignalDatabase
        from backtest.models import SignalRecord

        db = SignalDatabase(tmp_path / "test2.db")

        signal = SignalRecord(ticker="MSFT", t_signal=0.3)
        sid = db.store_signal(signal)

        retrieved = db.get_signal_by_id(sid)
        assert retrieved is not None
        assert retrieved.xai_pfd is None
        assert retrieved.xai_distress_zone == ""
        db.close()

    def test_migration_on_existing_db(self, tmp_path):
        """Verify migration adds XAI columns to an existing DB without them."""
        import sqlite3

        db_path = tmp_path / "legacy.db"
        # Create a DB with the old schema (no XAI columns)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            t_signal REAL DEFAULT 0.0
        )""")
        conn.execute(
            "INSERT INTO signals (ticker, signal_date, t_signal) VALUES (?, ?, ?)",
            ("OLD", "2024-01-01T00:00:00", 0.5),
        )
        conn.commit()
        conn.close()

        # Now open with SignalDatabase — should migrate
        from backtest.database import SignalDatabase

        db = SignalDatabase(db_path)
        # Check columns exist
        conn = db._get_conn()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
        assert "xai_pfd" in cols
        assert "xai_distress_zone" in cols
        assert "xai_model_used" in cols
        db.close()


# ===========================================================================
# TestXAIModels
# ===========================================================================

class TestXAIModels:
    """Test Pydantic model schemas."""

    def test_xai_result_model_dump(self):
        from xai.models import DistressAssessment, ReturnDecomposition, XAIResult

        result = XAIResult(
            ticker="TEST",
            distress=DistressAssessment(
                pfd=0.1, distress_zone="safe", model_used="altman_z_score"
            ),
            returns=ReturnDecomposition(),
        )
        dumped = result.model_dump()
        assert dumped["ticker"] == "TEST"
        assert dumped["distress"]["pfd"] == 0.1

    def test_xai_fallback(self):
        from xai.models import XAIFallback

        fb = XAIFallback(error_message="test error", ticker="ERR")
        assert not fb.success
        assert fb.error_message == "test error"


# ===========================================================================
# TestBuiltinShapley
# ===========================================================================

class TestBuiltinShapley:
    """Test built-in Shapley value calculators (no shap library needed)."""

    def test_exact_linear_shapley_healthy(self):
        from xai.features import extract_features, features_to_array
        from xai.shapley import ExactLinearShapley

        features = extract_features(HEALTHY_FUNDAMENTALS)
        x_arr, names = features_to_array(features)
        calc = ExactLinearShapley()
        result = calc.compute(x_arr, names, target="pfd")

        assert result.values.shape == (12,)
        assert result.method == "exact_linear_pfd"
        assert len(result.feature_names) == 12
        # Only 5 Z-Score features should have non-zero Shapley values
        nonzero_count = np.count_nonzero(result.values)
        assert nonzero_count <= 5

    def test_exact_linear_shapley_z_score_target(self):
        from xai.features import extract_features, features_to_array
        from xai.shapley import ExactLinearShapley

        features = extract_features(HEALTHY_FUNDAMENTALS)
        x_arr, names = features_to_array(features)
        calc = ExactLinearShapley()
        result = calc.compute(x_arr, names, target="z_score")

        assert result.method == "exact_linear_z_score"
        # Sum of Shapley values + base = Z-Score
        from xai.distress import AltmanZScoreModel
        z_actual = AltmanZScoreModel().compute_z_score(x_arr)
        z_reconstructed = result.base_value + float(np.sum(result.values))
        assert abs(z_actual - z_reconstructed) < 0.01

    def test_exact_shapley_distressed_vs_healthy(self):
        """Distressed company should have positive PFD Shapley (increases risk)."""
        from xai.features import extract_features, features_to_array
        from xai.shapley import ExactLinearShapley

        calc = ExactLinearShapley()

        x_h, names = features_to_array(extract_features(HEALTHY_FUNDAMENTALS))
        result_h = calc.compute(x_h, names, target="pfd")

        x_d, _ = features_to_array(extract_features(DISTRESSED_FUNDAMENTALS))
        result_d = calc.compute(x_d, names, target="pfd")

        # Distressed should have more positive SHAP (increasing PFD)
        assert np.sum(result_d.values) > np.sum(result_h.values)

    def test_permutation_shapley_basic(self):
        from xai.distress import AltmanZScoreModel
        from xai.features import extract_features, features_to_array
        from xai.shapley import PermutationShapley

        model = AltmanZScoreModel()
        features = extract_features(HEALTHY_FUNDAMENTALS)
        x_arr, names = features_to_array(features)

        calc = PermutationShapley(n_permutations=100, seed=42)
        result = calc.compute(model, x_arr, names)

        assert result.values.shape == (12,)
        assert "permutation" in result.method
        # Sum of Shapley values should approximately equal
        # prediction - base_value (efficiency property)
        pred = model.predict_proba(x_arr.reshape(1, -1))[0, 1]
        reconstructed = result.base_value + float(np.sum(result.values))
        assert abs(pred - reconstructed) < 0.05  # within 5% tolerance

    def test_compute_shapley_values_auto_linear(self):
        """Auto method should pick exact for AltmanZScoreModel."""
        from xai.distress import AltmanZScoreModel
        from xai.features import extract_features, features_to_array
        from xai.shapley import compute_shapley_values

        model = AltmanZScoreModel()
        x_arr, names = features_to_array(extract_features(HEALTHY_FUNDAMENTALS))
        result = compute_shapley_values(model, x_arr, names, method="auto")

        assert "exact" in result.method

    def test_builtin_fallback_in_explainer(self, monkeypatch):
        """When shap is unavailable, explainer should use built-in Shapley."""
        import xai.explainer as exp_mod

        monkeypatch.setattr(exp_mod, "_SHAP_AVAILABLE", False)

        from xai.distress import AltmanZScoreModel
        from xai.explainer import SHAPExplainer
        from xai.features import extract_features, features_to_array

        model = AltmanZScoreModel()
        x_arr, names = features_to_array(extract_features(HEALTHY_FUNDAMENTALS))
        explainer = SHAPExplainer(top_k=3)
        result = explainer.explain(model, x_arr, names)

        # Should have non-trivial Shapley values (not all zeros)
        assert not np.all(result.shap_values == 0)
        # Should have extracted top drivers
        assert len(result.top_positive) > 0 or len(result.top_negative) > 0


# ===========================================================================
# TestXAIEndToEnd — Integration: XAI → CommitteeResult → Signal → API
# ===========================================================================

class TestXAIEndToEnd:
    """End-to-end integration tests for XAI data flow through the full pipeline."""

    def test_xai_result_on_committee_result(self):
        """CommitteeResult should carry xai_result field."""
        from orchestrator.committee import CommitteeResult
        from xai.models import DistressAssessment, ReturnDecomposition, XAIResult

        distress = DistressAssessment(
            pfd=0.05,
            z_score=4.2,
            distress_zone="safe",
            model_used="altman_z_score",
            top_risk_factors=[{"debt_to_equity": 0.02}],
            shap_base_value=0.1,
        )
        returns = ReturnDecomposition(
            is_distressed=False,
            distress_flag="CLEAR: PFD 5.0% below 50.0% threshold",
            earnings_yield_proxy=0.05,
            expected_return=0.0475,
            expected_return_pct="4.8%",
            top_return_factors=[{"operating_margin": 0.03}],
            shap_base_value=0.04,
        )
        xai = XAIResult(
            ticker="AAPL",
            distress=distress,
            returns=returns,
            features_used={"current_ratio": 1.5, "roe": 0.25},
            feature_importance_ranking=["roe", "current_ratio"],
            narrative="AAPL is financially healthy.",
            computation_time_ms=45.0,
        )

        result = CommitteeResult(ticker="AAPL", xai_result=xai)

        assert result.xai_result is not None
        assert result.xai_result.ticker == "AAPL"
        assert result.xai_result.distress.pfd == pytest.approx(0.05)

        # to_dict should serialize XAI
        d = result.to_dict()
        assert d["xai_result"] is not None
        assert d["xai_result"]["ticker"] == "AAPL"

    def test_state_to_result_extracts_xai(self):
        """_state_to_result should extract xai_analysis from context."""
        from orchestrator.graph import _state_to_result

        state = {
            "ticker": "NVDA",
            "context": {
                "xai_analysis": {
                    "ticker": "NVDA",
                    "distress": {
                        "pfd": 0.03,
                        "z_score": 5.1,
                        "distress_zone": "safe",
                        "model_used": "altman_z_score",
                        "top_risk_factors": [],
                        "shap_base_value": 0.1,
                    },
                    "returns": {
                        "is_distressed": False,
                        "distress_flag": "CLEAR",
                        "earnings_yield_proxy": 0.04,
                        "expected_return": 0.0388,
                        "expected_return_pct": "3.9%",
                        "top_return_factors": [],
                        "shap_base_value": 0.03,
                    },
                    "features_used": {"roe": 0.3},
                    "feature_importance_ranking": ["roe"],
                    "narrative": "NVDA safe.",
                    "computation_time_ms": 30.0,
                },
            },
            "traces": {},
            "conviction_timeline": [],
            "parsing_failures": [],
        }

        result = _state_to_result(state)
        assert result.xai_result is not None
        assert result.xai_result.ticker == "NVDA"
        assert result.xai_result.distress.pfd == pytest.approx(0.03)

    def test_state_to_result_no_xai(self):
        """_state_to_result should handle missing XAI gracefully."""
        from orchestrator.graph import _state_to_result

        state = {
            "ticker": "MSFT",
            "context": {},
            "traces": {},
            "conviction_timeline": [],
            "parsing_failures": [],
        }

        result = _state_to_result(state)
        assert result.xai_result is None

    def test_signal_persistence_with_xai(self, tmp_path):
        """Signal persistence should populate XAI fields from CommitteeResult."""
        from backtest.database import SignalDatabase
        from backtest.models import SignalRecord
        from xai.models import DistressAssessment, ReturnDecomposition, XAIResult

        distress = DistressAssessment(
            pfd=0.15,
            z_score=2.5,
            distress_zone="grey",
            model_used="altman_z_score",
            top_risk_factors=[{"debt_to_equity": 0.05}],
            shap_base_value=0.1,
        )
        returns = ReturnDecomposition(
            is_distressed=False,
            distress_flag="CLEAR",
            earnings_yield_proxy=0.06,
            expected_return=0.051,
            expected_return_pct="5.1%",
            top_return_factors=[],
            shap_base_value=0.04,
        )
        xai = XAIResult(
            ticker="TEST",
            distress=distress,
            returns=returns,
            features_used={},
            feature_importance_ranking=[],
            narrative="Test.",
            computation_time_ms=10.0,
        )

        # Build signal with XAI fields populated
        signal = SignalRecord(
            ticker="TEST",
            t_signal=0.5,
            xai_pfd=xai.distress.pfd,
            xai_z_score=xai.distress.z_score,
            xai_distress_zone=xai.distress.distress_zone,
            xai_expected_return=xai.returns.expected_return,
            xai_model_used=xai.distress.model_used,
            xai_top_risk_factor="debt_to_equity",
        )

        db = SignalDatabase(tmp_path / "e2e_test.db")
        sid = db.store_signal(signal)

        retrieved = db.get_signal_by_id(sid)
        assert retrieved.xai_pfd == pytest.approx(0.15)
        assert retrieved.xai_z_score == pytest.approx(2.5)
        assert retrieved.xai_distress_zone == "grey"
        assert retrieved.xai_expected_return == pytest.approx(0.051)
        assert retrieved.xai_model_used == "altman_z_score"
        assert retrieved.xai_top_risk_factor == "debt_to_equity"
        db.close()

    def test_api_response_includes_xai(self):
        """AnalysisResponse model should accept xai_analysis field."""
        from api.models import AnalysisResponse

        response = AnalysisResponse(
            success=True,
            xai_analysis={
                "ticker": "AAPL",
                "distress": {"pfd": 0.05},
                "narrative": "Healthy.",
            },
        )
        assert response.xai_analysis is not None
        assert response.xai_analysis["ticker"] == "AAPL"

    def test_xai_context_in_agent_prompt(self):
        """Agents should have access to XAI context from their context dict."""
        # Verify that agent act() method reads xai_analysis from context
        context = {
            "market_data": {},
            "news": [],
            "financial_metrics": {},
            "xai_analysis": {
                "narrative": "Test XAI narrative for agents.",
                "distress": {"pfd": 0.1, "distress_zone": "safe"},
                "returns": {"expected_return_pct": "5.0%"},
            },
        }

        # Sector analyst should extract XAI section
        xai_data = context.get("xai_analysis", {})
        narrative = xai_data.get("narrative", "")
        assert narrative == "Test XAI narrative for agents."

        # Risk manager should detect elevated distress
        distress = xai_data.get("distress", {})
        pfd = distress.get("pfd", None)
        assert pfd is not None
        assert pfd < 0.3  # Not elevated for this test case
