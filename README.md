# weight-transfer

ParamΔ (Parameter Delta) の実装：モデル間の重みの差分を抽出して別のモデルに適用するツールです。

## 概要

このツールは、論文 ["ParamΔ for Direct Weight Mixing: Post-Train Large Language Model at Zero Cost" (arXiv:2504.21023)](https://arxiv.org/abs/2504.21023) の再現実装です。

ポストトレーニング済みモデルとベースモデルの重みの差分を計算し、その差分を新しいベースモデルに適用することで、追加の学習コストなしでポストトレーニングの知識を転移させることができます。

数式：`Θ_ParamΔ = Θ_post - Θ_base + Θ'_base`

## インストール

```bash
# Python 3.10以上が必要
# uvを使用する場合
uv pip install -e .

# または、uvの仮想環境を作成してインストール
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
uv pip install -e .
```

## 使い方

```python
from main import apply_diff

apply_diff(
    path_source="Qwen/Qwen2.5-1.5B-Instruct",  # ポストトレーニング済みモデル (Θ_post)
    path_base="Qwen/Qwen2.5-1.5B",             # ベースモデル (Θ_base)
    path_target="Qwen/Qwen2.5-Math-1.5B",      # 新しいベースモデル (Θ'_base)
    path_output="./output_model",               # 出力先ディレクトリ
    alpha=1.0,                                  # 差分のスケール係数
    device="cuda"                               # 使用するデバイス
)
```

## 動作原理

1. ポストトレーニング済みモデル（Θ_post）とベースモデル（Θ_base）の重みの差分を計算
2. その差分を新しいベースモデル（Θ'_base）に適用：`Θ'_base + α × (Θ_post - Θ_base)`
3. 結果を新しいモデルとして保存

これにより、追加の学習なしでポストトレーニングの能力を新しいベースモデルに転移させることができます。
