"""Smoke tests using synthetic data – no real data files required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TRAIN = 60
N_TEST = 20
N_FEATURES = 50  # simulated wavelength channels


def _make_synthetic_data(tmp: Path) -> None:
    """Write synthetic train.csv and test.csv into *tmp*."""
    rng = np.random.RandomState(0)
    wl_cols = [f"wl_{i}" for i in range(N_FEATURES)]

    X_train = rng.randn(N_TRAIN, N_FEATURES)
    y_train = X_train[:, 0] * 2 + rng.randn(N_TRAIN) * 0.1
    df_train = pd.DataFrame(X_train, columns=wl_cols)
    df_train.insert(0, "id", range(N_TRAIN))
    df_train["y"] = y_train
    df_train.to_csv(tmp / "train.csv", index=False)

    X_test = rng.randn(N_TEST, N_FEATURES)
    df_test = pd.DataFrame(X_test, columns=wl_cols)
    df_test.insert(0, "id", range(N_TEST))
    df_test.to_csv(tmp / "test.csv", index=False)


@pytest.fixture()
def synthetic_env(tmp_path):
    """Create a temporary environment with synthetic data and a config."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    _make_synthetic_data(data_dir)

    run_dir = tmp_path / "runs" / "smoke"
    sub_dir = tmp_path / "submissions"

    return {"data_dir": data_dir, "run_dir": run_dir, "sub_dir": sub_dir, "tmp": tmp_path}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreprocessPipeline:
    def test_snv_transform(self):
        from spectral_challenge.preprocess.snv import SNVTransformer

        X = np.random.randn(10, 30)
        Xt = SNVTransformer().fit_transform(X)
        # Each row should have mean ≈ 0, std ≈ 1
        assert np.allclose(Xt.mean(axis=1), 0, atol=1e-10)
        assert np.allclose(Xt.std(axis=1), 1, atol=1e-10)

    def test_msc_transform(self):
        from spectral_challenge.preprocess.msc import MSCTransformer

        X = np.random.randn(10, 30) + 5
        msc = MSCTransformer()
        Xt = msc.fit_transform(X)
        assert Xt.shape == X.shape

    def test_sg_transform(self):
        from spectral_challenge.preprocess.sg import SavitzkyGolayTransformer

        X = np.random.randn(10, 30)
        sg = SavitzkyGolayTransformer(window_length=7, polyorder=2, deriv=0)
        Xt = sg.fit_transform(X)
        assert Xt.shape == X.shape

    def test_derivative_transform(self):
        from spectral_challenge.preprocess.sg import DerivativeTransformer

        X = np.random.randn(10, 30)
        Xt = DerivativeTransformer(order=1).fit_transform(X)
        assert Xt.shape == (10, 29)

    def test_build_pipeline(self):
        from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

        steps = [
            {"name": "snv"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 0},
            {"name": "standard_scaler"},
        ]
        pipe = build_preprocess_pipeline(steps)
        X = np.random.randn(20, 30)
        Xt = pipe.fit_transform(X)
        assert Xt.shape == X.shape


class TestAbsorbanceTransformer:
    def test_shape_preserved(self):
        from spectral_challenge.preprocess.absorbance import AbsorbanceTransformer

        X = np.abs(np.random.randn(10, 30)) + 0.1
        Xt = AbsorbanceTransformer().fit_transform(X)
        assert Xt.shape == X.shape

    def test_monotonicity(self):
        """Larger input → smaller absorbance (since A = -log(X))."""
        from spectral_challenge.preprocess.absorbance import AbsorbanceTransformer

        X = np.array([[0.1, 0.5, 1.0]])
        Xt = AbsorbanceTransformer().fit_transform(X)
        # absorbance should decrease as reflectance increases
        assert Xt[0, 0] > Xt[0, 1] > Xt[0, 2]

    def test_base10(self):
        from spectral_challenge.preprocess.absorbance import AbsorbanceTransformer

        X = np.array([[10.0]])
        Xt = AbsorbanceTransformer(base="10").fit_transform(X)
        np.testing.assert_almost_equal(Xt[0, 0], -1.0)  # -log10(10) = -1

    def test_eps_clamp(self):
        """Zeros and negatives should be clamped, not produce -inf/NaN."""
        from spectral_challenge.preprocess.absorbance import AbsorbanceTransformer

        X = np.array([[0.0, -1.0, 1.0]])
        Xt = AbsorbanceTransformer().fit_transform(X)
        assert np.all(np.isfinite(Xt))


class TestMSCRobust:
    def test_msc_corrects_linear_scatter(self):
        """MSC should bring linearly scattered spectra close to reference."""
        from spectral_challenge.preprocess.msc import MSCTransformer

        rng = np.random.RandomState(42)
        ref = rng.randn(50) + 5
        # Create scattered copies: x_i = a_i + b_i * ref
        X = np.array([0.5 + 2.0 * ref, -1.0 + 0.8 * ref, 0.0 + 1.0 * ref])
        msc = MSCTransformer()
        msc.fit(X)
        Xt = msc.transform(X)
        # After correction, all rows should be close to the reference
        for i in range(Xt.shape[0]):
            corr = np.corrcoef(Xt[i], msc.reference_)[0, 1]
            assert corr > 0.999

    def test_msc_shape(self):
        from spectral_challenge.preprocess.msc import MSCTransformer

        X = np.random.randn(10, 30) + 5
        Xt = MSCTransformer().fit_transform(X)
        assert Xt.shape == X.shape


class TestEMSCTransformer:
    def test_emsc_shape(self):
        from spectral_challenge.preprocess.msc import EMSCTransformer

        X = np.random.randn(10, 30) + 5
        Xt = EMSCTransformer().fit_transform(X)
        assert Xt.shape == X.shape

    def test_emsc_removes_baseline(self):
        """EMSC should remove polynomial baseline drift."""
        from spectral_challenge.preprocess.msc import EMSCTransformer

        rng = np.random.RandomState(42)
        ref = rng.randn(50) + 5
        w = np.linspace(-1, 1, 50)
        # Add a quadratic baseline to a scaled ref
        X = np.array([
            1.5 * ref + 0.3 * w**2 + 0.1,
            0.8 * ref - 0.2 * w + 0.5,
            1.0 * ref,
        ])
        emsc = EMSCTransformer(poly_order=2)
        emsc.fit(X)
        Xt = emsc.transform(X)
        for i in range(Xt.shape[0]):
            corr = np.corrcoef(Xt[i], emsc.reference_)[0, 1]
            assert corr > 0.99


class TestBinningTransformer:
    def test_output_dimension(self):
        from spectral_challenge.preprocess.binning import BinningTransformer

        X = np.random.randn(10, 50)
        Xt = BinningTransformer(bin_size=4).fit_transform(X)
        # 50 // 4 = 12 full bins + 1 remainder bin (2 channels) = 13
        assert Xt.shape == (10, 13)

    def test_exact_division(self):
        from spectral_challenge.preprocess.binning import BinningTransformer

        X = np.random.randn(10, 40)
        Xt = BinningTransformer(bin_size=4).fit_transform(X)
        assert Xt.shape == (10, 10)

    def test_bin_values(self):
        """Each bin should be the mean of its constituent channels."""
        from spectral_challenge.preprocess.binning import BinningTransformer

        X = np.arange(8).reshape(1, 8).astype(float)
        Xt = BinningTransformer(bin_size=4).fit_transform(X)
        # bin 0 = mean(0,1,2,3) = 1.5, bin 1 = mean(4,5,6,7) = 5.5
        np.testing.assert_array_equal(Xt, [[1.5, 5.5]])


class TestNaNInfGuard:
    def test_nan_raises(self):
        from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

        steps = [{"name": "standard_scaler"}]
        pipe = build_preprocess_pipeline(steps)
        # Fit on clean data first
        pipe.fit(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        # Then transform data containing NaN
        X_bad = np.array([[1.0, float("nan"), 3.0]])
        with pytest.raises(ValueError, match="NaN/inf detected"):
            pipe.transform(X_bad)


class TestPipelineWithNewTransformers:
    def test_full_pipeline_absorbance_msc_sg(self):
        from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

        steps = [
            {"name": "absorbance"},
            {"name": "msc"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "standard_scaler"},
        ]
        pipe = build_preprocess_pipeline(steps)
        X = np.abs(np.random.randn(20, 30)) + 0.1
        Xt = pipe.fit_transform(X)
        assert Xt.shape == X.shape
        assert np.all(np.isfinite(Xt))

    def test_pipeline_with_binning(self):
        from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

        steps = [
            {"name": "binning", "bin_size": 5},
            {"name": "standard_scaler"},
        ]
        pipe = build_preprocess_pipeline(steps)
        X = np.random.randn(20, 50)
        Xt = pipe.fit_transform(X)
        assert Xt.shape == (20, 10)

    def test_pipeline_with_emsc(self):
        from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

        steps = [
            {"name": "emsc", "poly_order": 2},
            {"name": "standard_scaler"},
        ]
        pipe = build_preprocess_pipeline(steps)
        X = np.random.randn(20, 30) + 5
        Xt = pipe.fit_transform(X)
        assert Xt.shape == X.shape


class TestConfig:
    def test_from_yaml(self, tmp_path):
        from spectral_challenge.config import Config

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model_type: svr\nn_folds: 3\nseed: 123\n")
        cfg = Config.from_yaml(cfg_path)
        assert cfg.model_type == "svr"
        assert cfg.n_folds == 3
        assert cfg.seed == 123

    def test_overrides(self, tmp_path):
        from spectral_challenge.config import Config

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model_type: ridge\n")
        cfg = Config.from_yaml(cfg_path, overrides={"seed": 999})
        assert cfg.seed == 999


class TestModelFactory:
    def test_create_ridge(self):
        from spectral_challenge.models.factory import create_model

        m = create_model("ridge", {"alpha": 0.5})
        assert hasattr(m, "fit")

    def test_unknown_raises(self):
        from spectral_challenge.models.factory import create_model

        with pytest.raises(ValueError):
            create_model("nonexistent_model")


class TestConfigOverride:
    """Tests for Hydra-style --override parsing and application."""

    def test_parse_simple(self):
        from spectral_challenge.config_override import parse_overrides

        result = parse_overrides(["n_folds=3", "shuffle=false", "model_params.alpha=0.5"])
        assert result[0] == ("n_folds", ["n_folds"], 3)
        assert result[1] == ("shuffle", ["shuffle"], False)
        assert result[2] == ("model_params.alpha", ["model_params", "alpha"], 0.5)

    def test_parse_array_index(self):
        from spectral_challenge.config_override import parse_overrides

        result = parse_overrides(["preprocess[0].deriv=1"])
        assert result[0][1] == ["preprocess", 0, "deriv"]
        assert result[0][2] == 1

    def test_apply_nested(self):
        from spectral_challenge.config_override import apply_overrides, parse_overrides

        cfg = {"model_type": "ridge", "model_params": {"alpha": 1.0}}
        parsed = parse_overrides(["model_params.alpha=0.5"])
        apply_overrides(cfg, parsed)
        assert cfg["model_params"]["alpha"] == 0.5

    def test_apply_array(self):
        from spectral_challenge.config_override import apply_overrides, parse_overrides

        cfg = {"preprocess": [{"name": "snv"}, {"name": "sg", "deriv": 0}]}
        parsed = parse_overrides(["preprocess[1].deriv=2"])
        apply_overrides(cfg, parsed)
        assert cfg["preprocess"][1]["deriv"] == 2

    def test_invalid_key_raises(self):
        from spectral_challenge.config_override import apply_overrides, parse_overrides

        parsed = parse_overrides(["nonexistent_key=42"])
        with pytest.raises(ValueError, match="Unknown config key"):
            apply_overrides({}, parsed, valid_top_keys={"model_type", "seed"})

    def test_invalid_format_raises(self):
        from spectral_challenge.config_override import parse_overrides

        with pytest.raises(ValueError, match="Expected 'key=value'"):
            parse_overrides(["no_equals_sign"])

    def test_type_inference(self):
        from spectral_challenge.config_override import parse_overrides

        result = parse_overrides([
            "a=true", "b=123", "c=0.5", "d=null", "e=[1,2,3]", "f=hello",
        ])
        values = [r[2] for r in result]
        assert values == [True, 123, 0.5, None, [1, 2, 3], "hello"]

    def test_config_from_yaml_with_cli_overrides(self, tmp_path):
        from spectral_challenge.config import Config

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model_type: ridge\nmodel_params:\n  alpha: 1.0\nn_folds: 5\n")
        cfg = Config.from_yaml(cfg_path, cli_overrides=["model_params.alpha=0.5", "n_folds=3"])
        assert cfg.model_params["alpha"] == 0.5
        assert cfg.n_folds == 3

    def test_cli_cv_with_override(self, synthetic_env, tmp_path):
        """End-to-end: cv command with --override."""
        from spectral_challenge.cli import main

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "model_type: ridge\n"
            "model_params:\n  alpha: 1.0\n"
            "preprocess:\n  - name: standard_scaler\n"
            "n_folds: 3\n"
        )
        run_dir = synthetic_env["run_dir"]
        data_dir = synthetic_env["data_dir"]

        main([
            "cv",
            "--config", str(cfg_path),
            "--outdir", str(run_dir),
            "--data-dir", str(data_dir),
            "--override", "model_params.alpha=0.01",
            "--set", "seed=0",
        ])

        assert (run_dir / "metrics.json").exists()

    def test_cli_cv_override_preprocess_to_absorbance(self, synthetic_env, tmp_path):
        """E2E: swap preprocess step via --override preprocess[0].name=..."""
        from spectral_challenge.cli import main

        # Synthetic data needs to be positive for absorbance
        data_dir = tmp_path / "data_pos"
        data_dir.mkdir()
        rng = np.random.RandomState(0)
        wl_cols = [f"wl_{i}" for i in range(N_FEATURES)]
        X_train = np.abs(rng.randn(N_TRAIN, N_FEATURES)) + 0.1
        y_train = X_train[:, 0] * 2 + rng.randn(N_TRAIN) * 0.1
        df_train = pd.DataFrame(X_train, columns=wl_cols)
        df_train.insert(0, "id", range(N_TRAIN))
        df_train["y"] = y_train
        df_train.to_csv(data_dir / "train.csv", index=False)

        cfg_path = tmp_path / "cfg_abs.yaml"
        cfg_path.write_text(
            "model_type: ridge\n"
            "model_params:\n  alpha: 1.0\n"
            "preprocess:\n  - name: standard_scaler\n"
            "n_folds: 3\n"
        )
        run_dir = tmp_path / "runs" / "abs_override"

        main([
            "cv",
            "--config", str(cfg_path),
            "--outdir", str(run_dir),
            "--data-dir", str(data_dir),
            "--override", "preprocess[0].name=absorbance",
        ])

        assert (run_dir / "metrics.json").exists()


class TestColumnAutoDetection:
    """Tests for robust column auto-detection in data loading."""

    def test_id_col_auto_detect_sample_number(self, tmp_path):
        """When config says id_col='id' but CSV has 'sample number'."""
        from spectral_challenge.config import Config
        from spectral_challenge.data.load import load_train

        rng = np.random.RandomState(0)
        df = pd.DataFrame({"sample number": range(10), "含水率": rng.randn(10)})
        for i in range(5):
            df[f"{9000 + i}.123"] = rng.randn(10)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df.to_csv(data_dir / "train.csv", index=False)

        cfg = Config(id_col="id", target_col="含水率")
        X, y, ids = load_train(cfg, data_dir)
        assert list(ids) == list(range(10))
        assert X.shape == (10, 5)

    def test_target_col_auto_detect_moisture(self, tmp_path):
        """When config says target_col='y' but CSV has '含水率'."""
        from spectral_challenge.config import Config
        from spectral_challenge.data.load import load_train

        rng = np.random.RandomState(0)
        target = rng.randn(10)
        df = pd.DataFrame({"id": range(10), "含水率": target, "feat_a": rng.randn(10)})
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df.to_csv(data_dir / "train.csv", index=False)

        cfg = Config(id_col="id", target_col="y")
        X, y, ids = load_train(cfg, data_dir)
        np.testing.assert_array_almost_equal(y, target)

    def test_normalised_column_matching(self, tmp_path):
        """Whitespace/case differences should still match."""
        from spectral_challenge.config import Config
        from spectral_challenge.data.load import load_train

        rng = np.random.RandomState(0)
        # Column with leading/trailing spaces and different case
        df = pd.DataFrame({" Sample Number ": range(10), "Y": rng.randn(10), "f": rng.randn(10)})
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df.to_csv(data_dir / "train.csv", index=False)

        cfg = Config(id_col="sample number", target_col="y")
        X, y, ids = load_train(cfg, data_dir)
        assert len(ids) == 10

    def test_float_column_name_features(self, tmp_path):
        """Columns named as float wavelengths should be auto-detected as features."""
        from spectral_challenge.config import Config
        from spectral_challenge.data.load import load_train

        rng = np.random.RandomState(0)
        n_wl = 20
        data = {"sample number": range(10), "species number": [1] * 10, "含水率": rng.randn(10)}
        for i in range(n_wl):
            data[f"{9000 + i * 10}.12345"] = rng.randn(10)
        df = pd.DataFrame(data)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df.to_csv(data_dir / "train.csv", index=False)

        cfg = Config(id_col="sample number", target_col="含水率")
        X, y, ids = load_train(cfg, data_dir)
        # 'species number' is NOT a float name → excluded; only wavelength cols
        assert X.shape == (10, n_wl)

    def test_missing_target_raises(self, tmp_path):
        """Should raise KeyError when target column cannot be found at all."""
        from spectral_challenge.config import Config
        from spectral_challenge.data.load import load_train

        df = pd.DataFrame({"id": range(5), "feat": [1.0] * 5})
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df.to_csv(data_dir / "train.csv", index=False)

        cfg = Config(id_col="id", target_col="nonexistent")
        with pytest.raises(KeyError, match="Could not detect target column"):
            load_train(cfg, data_dir)

    def test_id_fallback_first_column(self, tmp_path):
        """If no id candidates match, fall back to the first column."""
        from spectral_challenge.config import Config
        from spectral_challenge.data.load import load_train

        rng = np.random.RandomState(0)
        df = pd.DataFrame({"番号": range(10), "y": rng.randn(10), "feat": rng.randn(10)})
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df.to_csv(data_dir / "train.csv", index=False)

        cfg = Config(id_col="id", target_col="y")
        X, y, ids = load_train(cfg, data_dir)
        # Should fall back to first column '番号'
        assert list(ids) == list(range(10))


class TestEndToEnd:
    """Full pipeline: CV → predict → submit on synthetic data."""

    def test_cv_predict_submit(self, synthetic_env):
        from spectral_challenge.config import Config
        from spectral_challenge.data.load import load_test, load_train
        from spectral_challenge.predict import predict_test
        from spectral_challenge.submit import make_submission
        from spectral_challenge.train import run_cv

        cfg = Config(n_folds=3, seed=42)
        data_dir = synthetic_env["data_dir"]
        run_dir = synthetic_env["run_dir"]

        # CV
        X, y, ids = load_train(cfg, data_dir)
        result = run_cv(cfg, X, y, run_dir)
        assert "mean_rmse" in result
        assert result["mean_rmse"] > 0
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "oof_preds.npy").exists()

        # Predict
        X_test, test_ids = load_test(cfg, data_dir)
        preds = predict_test(X_test, run_dir)
        assert preds.shape == (N_TEST,)

        # Submit
        out = synthetic_env["sub_dir"] / "submission.csv"
        make_submission(test_ids, preds, cfg.id_col, cfg.target_col, out)
        assert out.exists()
        df_sub = pd.read_csv(out)
        assert list(df_sub.columns) == ["id", "y"]
        assert len(df_sub) == N_TEST

    def test_cli_cv(self, synthetic_env, tmp_path):
        """Test the CLI cv command end-to-end."""
        from spectral_challenge.cli import main

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "model_type: ridge\n"
            "model_params:\n  alpha: 1.0\n"
            "preprocess:\n  - name: standard_scaler\n"
            "n_folds: 3\n"
        )
        run_dir = synthetic_env["run_dir"]
        data_dir = synthetic_env["data_dir"]

        main(
            [
                "cv",
                "--config",
                str(cfg_path),
                "--outdir",
                str(run_dir),
                "--data-dir",
                str(data_dir),
            ]
        )

        assert (run_dir / "metrics.json").exists()
