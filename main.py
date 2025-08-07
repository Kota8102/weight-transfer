import torch
import tqdm
import transformers


@torch.inference_mode()
def apply_diff(
    path_source: str,
    path_base: str,
    path_target: str,
    path_output: str,
    alpha: float = 1.0,
    device: str = "cpu",
) -> None:
    """2つのモデル間の差分を第3のモデルに適用します。

    実行される操作: ``target += alpha × (source - base)``

    これにより、sourceモデルがbaseモデルから学習した変化を、
    別のtargetモデルに転移させることができます。

    Args:
        path_source: 差分の元となるモデルへのパス（例：ファインチューニング済みモデル）
        path_base: ベースモデルへのパス
        path_target: 差分を適用する対象モデルへのパス
        path_output: 出力モデルへのパス
        alpha: 差分のスケール係数
        device: デバイス（"cpu" または "cuda"）

    Returns:
        None

    """
    model_source = transformers.AutoModelForCausalLM.from_pretrained(
        path_source,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    model_base = transformers.AutoModelForCausalLM.from_pretrained(
        path_base,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    model_target = transformers.AutoModelForCausalLM.from_pretrained(
        path_target,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    print("Source model keys:", list(model_source.state_dict().keys())[:5], "...")
    print("Base model keys:", list(model_base.state_dict().keys())[:5], "...")
    print("Target model keys:", list(model_target.state_dict().keys())[:5], "...")

    # 状態辞書を取得
    sd_source = model_source.state_dict()
    sd_base = model_base.state_dict()
    sd_target = model_target.state_dict()

    # sourceとbaseの差分を計算
    sd_diff = {}
    for key in sd_source:
        if key in sd_base:
            sd_diff[key] = sd_source[key] - sd_base[key]
        else:
            raise KeyError(f"{key} missing from base checkpoint.")

    # targetモデルに差分を適用
    for key in tqdm.tqdm(sd_target, desc="Applying diff"):
        if key not in sd_diff:
            raise KeyError(f"{key} missing from diff checkpoint.")
        if sd_target[key].shape != sd_diff[key].shape:
            raise ValueError(
                f"Shape mismatch at {key}: {sd_target[key].shape} vs {sd_diff[key].shape}",
            )
        sd_target[key].add_(alpha * sd_diff[key])

    # 結果を保存
    print(f"Saving model to {path_output}")
    model_target.save_pretrained(path_output)

    # トークナイザーも保存（sourceから取得）
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(path_source)
        tokenizer.save_pretrained(path_output)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")


if __name__ == "__main__":
    apply_diff(
        path_source="Qwen/Qwen3-14B",  # 学習済みモデル
        path_base="Qwen/Qwen3-14B-Base",  # ベースモデル
        path_target="Kota8102/qwen3-14b-test",  # 適用先（別のモデルでも可）
        path_output="./output_model",  # 出力先
        device="mps",
    )
