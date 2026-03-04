# SIGNATE Spectrum Challenge

近赤外（NIR）スペクトルから含水率を回帰予測するパイプライン。評価指標は **RMSE**。

## ディレクトリ構成

```
├── configs/               # 実験設定 YAML
│   ├── baseline_ridge.yaml
│   ├── baseline_pls.yaml
│   └── baseline_lgbm.yaml
├── src/spectral_challenge/ # コアモジュール
│   ├── cli.py             # エントリポイント
│   ├── config.py          # YAML 設定読込
│   ├── data/              # データ読込・CV分割
│   ├── preprocess/        # 前処理（SNV, MSC, SG, 微分, StandardScaler）
│   ├── models/            # モデルファクトリ（Ridge, PLS, SVR, LightGBM）
│   ├── train.py           # CV学習ループ
│   ├── predict.py         # Fold平均推論
│   ├── submit.py          # submission CSV 生成
│   └── metrics.py         # RMSE 等
├── notebooks/             # 探索用ノートブック
├── tests/                 # smoke test
├── data/                  # ← Git管理外
│   └── raw/               # train.csv, test.csv を配置
├── runs/                  # ← Git管理外：実験結果
├── submissions/           # ← Git管理外：提出ファイル
├── Makefile
├── pyproject.toml
└── README.md
```

## セットアップ

```bash
# 1. venv 作成 & 有効化
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. パッケージインストール（編集モード + 開発ツール）
pip install -e ".[dev]"
```

### 依存ライブラリ（主要）

| パッケージ | 用途 |
|-----------|------|
| numpy, pandas | 数値処理・データ操作 |
| scikit-learn | 前処理・モデル・CV分割 |
| lightgbm | LightGBM |
| scipy | Savitzky-Golay フィルタ |
| pyyaml | 設定ファイル読込 |
| joblib | モデル永続化 |
| pytest, ruff | テスト・リンタ（dev） |

## データ配置

`data/raw/` に以下のファイルを配置してください：

| ファイル名 | 説明 |
|-----------|------|
| `train.csv` | 学習データ（ID列 + スペクトル特徴量 + 目的変数） |
| `test.csv` | テストデータ（ID列 + スペクトル特徴量） |

**列名の想定**（config で変更可能）:
- ID列: `id`
- 目的変数列: `y`
- 特徴量列: 上記以外の全数値列を自動検出（`feature_prefix` で絞り込みも可）

列名がデータと異なる場合は、config YAML の `id_col` / `target_col` を修正してください。

## 使い方

### 1コマンド CV 実行

```bash
# Ridge ベースライン
python -m spectral_challenge.cli cv --config configs/baseline_ridge.yaml

# PLS
python -m spectral_challenge.cli cv --config configs/baseline_pls.yaml

# LightGBM
python -m spectral_challenge.cli cv --config configs/baseline_lgbm.yaml
```

### Makefile 経由

```bash
make cv CONFIG=configs/baseline_ridge.yaml
```

### 推論 & 提出ファイル生成

```bash
# CV で生成された run ディレクトリを指定
python -m spectral_challenge.cli submit \
    --config configs/baseline_ridge.yaml \
    --run-dir runs/baseline_ridge_ridge_20260304_120000
```

### CLI オプション

| オプション | 説明 |
|-----------|------|
| `--config` | YAML 設定ファイルパス（必須） |
| `--seed` | 乱数シード上書き |
| `--folds` | CV fold 数上書き |
| `--outdir` | 出力ディレクトリ上書き |
| `--data-dir` | データディレクトリ上書き |

### サブコマンド

| コマンド | 説明 |
|---------|------|
| `cv` | K-Fold CV 学習 → OOF予測・RMSE・モデル保存 |
| `fit` | 全学習データで1モデル学習 |
| `predict` | テスト推論（fold 平均） |
| `submit` | テスト推論 → submission.csv 生成 |

## 実験出力 (`runs/`)

各 run ディレクトリに以下が保存されます：

```
runs/<experiment_name>/
├── config.yaml          # 使用した設定のコピー
├── git_hash.txt         # git commit hash
├── run.log              # 実行ログ
├── metrics.json         # fold別 & 平均 RMSE
├── oof_preds.npy        # OOF 予測
└── models/
    ├── model_fold0.joblib
    ├── pipe_fold0.joblib
    ├── ...
```

## 前処理パイプライン

config の `preprocess` セクションで前処理を順序付きリストとして定義します。
sklearn 互換（fit/transform）で、学習時に fit した統計量を推論時に再利用します。

利用可能な前処理:

| 名前 | 説明 | パラメータ例 |
|------|------|-------------|
| `snv` | Standard Normal Variate（行正規化） | なし |
| `msc` | Multiplicative Scatter Correction | なし |
| `sg` / `savitzky_golay` | Savitzky-Golay フィルタ | `window_length`, `polyorder`, `deriv` |
| `derivative` | 有限差分微分 | `order` |
| `standard_scaler` | 列方向の標準化 | なし |

設定例:
```yaml
preprocess:
  - name: snv
  - name: sg
    window_length: 11
    polyorder: 2
    deriv: 1
  - name: standard_scaler
```

## テスト

```bash
# smoke test（合成データ、実データ不要）
make test

# lint
make lint
```

## よくある失敗

| 症状 | 原因 | 対処 |
|------|------|------|
| `FileNotFoundError: train.csv` | データ未配置 | `data/raw/` にCSVを置く |
| `KeyError: 'y'` | 目的変数列名の不一致 | config の `target_col` を修正 |
| `KeyError: 'id'` | ID列名の不一致 | config の `id_col` を修正 |
| 特徴量が0列 | 数値列が見つからない | `feature_prefix` を確認、列名を確認 |
| NaN in predictions | 入力データにNaN | 前処理に imputation を追加するか、データを確認 |
| shape 不一致 | train と test の列数不一致 | CSV の列構成を確認 |

## 拡張のヒント

- **1D CNN / Transformer**: `src/spectral_challenge/models/factory.py` の `_REGISTRY` に追加。sklearn 互換の `fit`/`predict` インターフェースを持てばそのまま動く
- **GroupKFold**: config で `split_method: group_kfold`, `group_col: <列名>` を指定
- **新しい前処理**: `src/spectral_challenge/preprocess/` に sklearn 互換クラスを作り、`pipeline.py` の `_REGISTRY` に登録
