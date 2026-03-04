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
