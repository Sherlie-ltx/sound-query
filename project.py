# ============================================================
# 1. Imports
# ============================================================

import argparse
import hashlib
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer, ClapModel


# ============================================================
# 2. Configuration
# ============================================================

TARGET_SR = 44100
OUTPUT_SUBTYPE = "PCM_16"
MODEL_ID = "laion/clap-htsat-unfused"
CLAP_SR = 48000
DEFAULT_SEED = 42


def split_from_fold(fold):
    if fold in [1, 2, 3]:
        return "train"
    if fold == 4:
        return "val"
    if fold == 5:
        return "test"
    return "unknown"


def stable_seed(*parts):
    text = "|".join(str(p) for p in parts)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_load_any(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def choose_device(device):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 3. Audio Effects
# ============================================================

def db_to_amp(db):
    return 10 ** (db / 20)


def prevent_clipping(y, peak=0.98):
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y)
    max_abs = np.max(np.abs(y)) if len(y) else 0.0
    if max_abs > peak:
        y = y / max_abs * peak
    return y.astype(np.float32)


def load_audio(path, sr=TARGET_SR):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return prevent_clipping(y)


def apply_biquad(y, b, a):
    return prevent_clipping(signal.lfilter(b, a, y))


def high_shelf(y, sr, f0=4000, gain_db=6.0, s=0.8):
    a_gain = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cosw = np.cos(w0)
    sinw = np.sin(w0)
    alpha = sinw / 2 * np.sqrt((a_gain + 1 / a_gain) * (1 / s - 1) + 2)
    sqrt_a = np.sqrt(a_gain)

    b0 = a_gain * ((a_gain + 1) + (a_gain - 1) * cosw + 2 * sqrt_a * alpha)
    b1 = -2 * a_gain * ((a_gain - 1) + (a_gain + 1) * cosw)
    b2 = a_gain * ((a_gain + 1) + (a_gain - 1) * cosw - 2 * sqrt_a * alpha)
    a0 = (a_gain + 1) - (a_gain - 1) * cosw + 2 * sqrt_a * alpha
    a1 = 2 * ((a_gain - 1) - (a_gain + 1) * cosw)
    a2 = (a_gain + 1) - (a_gain - 1) * cosw - 2 * sqrt_a * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return apply_biquad(y, b, a)


def low_shelf(y, sr, f0=250, gain_db=4.0, s=0.8):
    a_gain = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cosw = np.cos(w0)
    sinw = np.sin(w0)
    alpha = sinw / 2 * np.sqrt((a_gain + 1 / a_gain) * (1 / s - 1) + 2)
    sqrt_a = np.sqrt(a_gain)

    b0 = a_gain * ((a_gain + 1) - (a_gain - 1) * cosw + 2 * sqrt_a * alpha)
    b1 = 2 * a_gain * ((a_gain - 1) - (a_gain + 1) * cosw)
    b2 = a_gain * ((a_gain + 1) - (a_gain - 1) * cosw - 2 * sqrt_a * alpha)
    a0 = (a_gain + 1) + (a_gain - 1) * cosw + 2 * sqrt_a * alpha
    a1 = -2 * ((a_gain - 1) + (a_gain + 1) * cosw)
    a2 = (a_gain + 1) + (a_gain - 1) * cosw - 2 * sqrt_a * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return apply_biquad(y, b, a)


def peaking_eq(y, sr, f0=300, gain_db=-6.0, q=1.0):
    a_gain = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    cosw = np.cos(w0)

    b0 = 1 + alpha * a_gain
    b1 = -2 * cosw
    b2 = 1 - alpha * a_gain
    a0 = 1 + alpha / a_gain
    a1 = -2 * cosw
    a2 = 1 - alpha / a_gain

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return apply_biquad(y, b, a)


def lowpass(y, sr, cutoff=3000, order=4):
    sos = signal.butter(order, cutoff, btype="lowpass", fs=sr, output="sos")
    return prevent_clipping(signal.sosfiltfilt(sos, y))


def bandpass(y, sr, low=250, high=4500, order=4):
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return prevent_clipping(signal.sosfiltfilt(sos, y))


def tanh_distortion(y, drive=3.0, mix=0.7):
    wet = np.tanh(drive * y) / np.tanh(drive)
    return prevent_clipping((1 - mix) * y + mix * wet)


def bitcrush(y, bits=8):
    levels = 2 ** bits
    return prevent_clipping(np.round(y * levels) / levels)


def add_noise(y, noise_db=-38, seed=0):
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    noise = rng.normal(0, rms * db_to_amp(noise_db), size=len(y)).astype(np.float32)
    return prevent_clipping(y + noise)


def simple_reverb(y, sr, wet=0.35, decay_time=0.75, seed=0):
    rng = np.random.default_rng(seed)
    ir_len = int(sr * decay_time)
    t = np.linspace(0, decay_time, ir_len, endpoint=False)
    decay = np.exp(-5.5 * t / decay_time)

    impulse = np.zeros(ir_len, dtype=np.float32)
    impulse[0] = 1.0
    for delay_ms, amp in [(18, 0.45), (43, 0.28), (71, 0.18)]:
        idx = int(sr * delay_ms / 1000)
        if idx < ir_len:
            impulse[idx] += amp

    impulse += 0.035 * rng.normal(0, 1, ir_len).astype(np.float32) * decay
    impulse = impulse / (np.max(np.abs(impulse)) + 1e-12)

    wet_sig = signal.fftconvolve(y, impulse, mode="full")[:len(y)]
    return prevent_clipping((1 - wet) * y + wet * wet_sig)


def chorus(y, sr, rate=0.35, depth_ms=8.0, base_delay_ms=18.0, mix=0.25):
    n = len(y)
    t = np.arange(n) / sr
    delay = (base_delay_ms + depth_ms * np.sin(2 * np.pi * rate * t)) * sr / 1000
    idx = np.arange(n) - delay
    delayed = np.interp(idx, np.arange(n), y, left=0.0, right=0.0).astype(np.float32)
    return prevent_clipping((1 - mix) * y + mix * delayed)


def soft_compress(y, threshold=0.28, ratio=3.0, makeup_db=1.5):
    sign = np.sign(y)
    mag = np.abs(y)
    above = mag > threshold
    compressed = mag.copy()
    compressed[above] = threshold + (mag[above] - threshold) / ratio
    return prevent_clipping(sign * compressed * db_to_amp(makeup_db))


def tremolo(y, sr, rate=5.5, depth=0.16):
    t = np.arange(len(y)) / sr
    modulation = 1.0 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
    return prevent_clipping(y * modulation)


def wow_flutter(y, sr, rate=0.45, depth_ms=3.5, mix=0.55):
    n = len(y)
    t = np.arange(n) / sr
    delay = depth_ms * np.sin(2 * np.pi * rate * t) * sr / 1000
    idx = np.arange(n) - delay
    shifted = np.interp(idx, np.arange(n), y, left=0.0, right=0.0).astype(np.float32)
    return prevent_clipping((1 - mix) * y + mix * shifted)


# ============================================================
# 4. Modifier Recipes
# ============================================================

def effect_brighter(y, sr, seed):
    return high_shelf(y, sr, f0=3500, gain_db=6.0)


def effect_more_muffled(y, sr, seed):
    y = lowpass(y, sr, cutoff=2600)
    y = high_shelf(y, sr, f0=4500, gain_db=-5.0)
    return prevent_clipping(y)


def effect_more_distorted(y, sr, seed):
    return tanh_distortion(y, drive=4.0, mix=0.75)


def effect_more_spacious(y, sr, seed):
    return simple_reverb(y, sr, wet=0.38, decay_time=0.85, seed=seed)


def effect_less_muddy(y, sr, seed):
    y = peaking_eq(y, sr, f0=320, gain_db=-7.0, q=1.0)
    y = high_shelf(y, sr, f0=4500, gain_db=1.5)
    return prevent_clipping(y)


def effect_more_cinematic(y, sr, seed):
    y = low_shelf(y, sr, f0=180, gain_db=4.0)
    y = peaking_eq(y, sr, f0=420, gain_db=-2.5, q=0.9)
    y = soft_compress(y, threshold=0.25, ratio=3.0, makeup_db=1.5)
    y = simple_reverb(y, sr, wet=0.34, decay_time=1.1, seed=seed)
    y = high_shelf(y, sr, f0=8500, gain_db=-1.5)
    return prevent_clipping(y)


def effect_more_dreamy(y, sr, seed):
    y = chorus(y, sr, rate=0.28, depth_ms=10.0, base_delay_ms=22.0, mix=0.28)
    y = simple_reverb(y, sr, wet=0.48, decay_time=1.15, seed=seed)
    y = lowpass(y, sr, cutoff=8500)
    return prevent_clipping(y)


def effect_more_lofi(y, sr, seed):
    y = bandpass(y, sr, low=220, high=4200)
    y = tanh_distortion(y, drive=2.6, mix=0.45)
    y = bitcrush(y, bits=8)
    y = add_noise(y, noise_db=-38, seed=seed)
    return prevent_clipping(y)


def effect_warmer(y, sr, seed):
    y = low_shelf(y, sr, f0=240, gain_db=3.5)
    y = peaking_eq(y, sr, f0=650, gain_db=1.5, q=0.8)
    y = high_shelf(y, sr, f0=6500, gain_db=-2.5)
    y = tanh_distortion(y, drive=1.6, mix=0.20)
    return prevent_clipping(y)


def effect_more_tense(y, sr, seed):
    y = peaking_eq(y, sr, f0=2200, gain_db=4.5, q=1.0)
    y = high_shelf(y, sr, f0=5200, gain_db=2.0)
    y = soft_compress(y, threshold=0.22, ratio=4.0, makeup_db=1.5)
    y = tanh_distortion(y, drive=2.3, mix=0.32)
    y = tremolo(y, sr, rate=5.2, depth=0.14)
    return prevent_clipping(y)


def effect_more_nostalgic(y, sr, seed):
    y = low_shelf(y, sr, f0=260, gain_db=2.5)
    y = peaking_eq(y, sr, f0=700, gain_db=1.2, q=0.8)
    y = high_shelf(y, sr, f0=5600, gain_db=-4.5)
    y = lowpass(y, sr, cutoff=9000)
    y = tanh_distortion(y, drive=1.6, mix=0.22)
    y = wow_flutter(y, sr, rate=0.42, depth_ms=3.0, mix=0.50)
    y = simple_reverb(y, sr, wet=0.16, decay_time=0.35, seed=seed)
    y = add_noise(y, noise_db=-43, seed=seed)
    return prevent_clipping(y)


def get_modifier_recipes(modifier_set):
    base_modifiers = [
        ("brighter", "L1", "brighter", "high_shelf_eq", effect_brighter),
        ("more muffled", "L1", "more_muffled", "low_pass_filter_plus_high_cut", effect_more_muffled),
        ("more distorted", "L1", "more_distorted", "tanh_saturation", effect_more_distorted),
        ("more spacious", "L1", "more_spacious", "synthetic_reverb", effect_more_spacious),
        ("less muddy", "L1", "less_muddy", "low_mid_cut_plus_slight_high_boost", effect_less_muddy),
        ("more cinematic", "L2", "more_cinematic", "low_boost_low_mid_cut_compression_reverb_soft_high_cut", effect_more_cinematic),
        ("more dreamy", "L2", "more_dreamy", "chorus_reverb_soft_high_cut", effect_more_dreamy),
        ("more lo-fi", "L2", "more_lofi", "bandpass_saturation_bitcrush_noise", effect_more_lofi),
        ("warmer", "L2", "warmer", "low_mid_boost_high_cut_mild_saturation", effect_warmer),
    ]
    l3_modifiers = [
        ("more tense", "L3", "more_tense", "high_mid_boost_compression_saturation_tremolo", effect_more_tense),
        ("more nostalgic", "L3", "more_nostalgic", "warm_eq_high_cut_tape_saturation_wow_flutter_noise_small_reverb", effect_more_nostalgic),
    ]

    selected = base_modifiers if str(modifier_set) == "9" else base_modifiers + l3_modifiers

    return [
        {
            "modifier": modifier,
            "modifier_level": level,
            "slug": slug,
            "effect_recipe": recipe,
            "function": function,
        }
        for modifier, level, slug, recipe, function in selected
    ]


# ============================================================
# 5. Dataset Builder
# ============================================================

def find_esc50_root(base):
    base = Path(base)
    matches = list(base.rglob("esc50.csv"))
    if not matches:
        return None
    return matches[0].parent.parent


def resolve_esc50_root(esc50_root=None, esc50_zip=None, extract_dir=None, overwrite=False):
    if esc50_root is not None:
        root = Path(esc50_root)
        if (root / "meta" / "esc50.csv").exists() and (root / "audio").exists():
            return root

    if esc50_zip is not None:
        zip_path = Path(esc50_zip)
        target_dir = Path(extract_dir) if extract_dir is not None else zip_path.parent / "esc50_extracted"
        if target_dir.exists() and overwrite:
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(target_dir)
        found = find_esc50_root(target_dir)
        if found is not None:
            return found

    search_base = Path.cwd() if esc50_root is None else Path(esc50_root)
    found = find_esc50_root(search_base)
    if found is not None:
        return found

    raise FileNotFoundError("ESC-50 root folder could not be found.")


def build_triplet_dataset(args):
    esc50_root = resolve_esc50_root(
        esc50_root=args.esc50_root,
        esc50_zip=args.esc50_zip,
        extract_dir=args.extract_dir,
        overwrite=args.overwrite,
    )
    output_root = Path(args.output_root)
    modifiers = get_modifier_recipes(args.modifier_set)

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    original_dir = output_root / "audio_original"
    processed_dir = output_root / "audio_processed"
    original_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for modifier in modifiers:
        (processed_dir / modifier["slug"]).mkdir(parents=True, exist_ok=True)

    meta_path = esc50_root / "meta" / "esc50.csv"
    audio_dir = esc50_root / "audio"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio folder: {audio_dir}")

    meta = pd.read_csv(meta_path)
    required_cols = {"filename", "fold", "target", "category"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"ESC-50 metadata missing columns: {sorted(missing)}")

    if args.max_source_files is not None:
        meta = meta.head(args.max_source_files).copy()

    rows = []
    failures = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Generating perceptual targets"):
        filename = row["filename"]
        fold = int(row["fold"])
        split = split_from_fold(fold)
        category = row["category"]
        esc50_target = int(row["target"])
        src_path = audio_dir / filename

        if not src_path.exists():
            failures.append((filename, "source_missing"))
            continue

        try:
            y = load_audio(src_path, TARGET_SR)
        except Exception as exc:
            failures.append((filename, f"load_error: {exc}"))
            continue

        if args.copy_originals:
            copied_src_path = original_dir / filename
            if args.overwrite or not copied_src_path.exists():
                shutil.copy2(src_path, copied_src_path)
            source_relpath = str(copied_src_path.relative_to(output_root))
        else:
            source_relpath = str(src_path)

        for modifier in modifiers:
            seed = stable_seed(filename, modifier["slug"])
            target_filename = f"{Path(filename).stem}__{modifier['slug']}.wav"
            target_path = processed_dir / modifier["slug"] / target_filename

            try:
                if args.overwrite or not target_path.exists():
                    y_target = modifier["function"](y.copy(), TARGET_SR, seed)
                    sf.write(target_path, y_target, TARGET_SR, subtype=OUTPUT_SUBTYPE)

                rows.append({
                    "triplet_id": f"{Path(filename).stem}__{modifier['slug']}",
                    "split": split,
                    "source_path": source_relpath,
                    "target_path": str(target_path.relative_to(output_root)),
                    "modifier": modifier["modifier"],
                    "modifier_level": modifier["modifier_level"],
                    "effect_recipe": modifier["effect_recipe"],
                    "modifier_slug": modifier["slug"],
                    "esc50_filename": filename,
                    "esc50_fold": fold,
                    "esc50_target": esc50_target,
                    "esc50_category": category,
                    "sample_rate": TARGET_SR,
                })
            except Exception as exc:
                failures.append((f"{filename}::{modifier['slug']}", f"effect_error: {exc}"))

    triplets = pd.DataFrame(rows)
    metadata_name = args.metadata_name or (
        "metadata_triplets_11mod.csv" if str(args.modifier_set) == "11" else "metadata_triplets.csv"
    )
    metadata_path = output_root / metadata_name
    triplets.to_csv(metadata_path, index=False)

    recipe_summary = [
        {
            "modifier": modifier["modifier"],
            "modifier_level": modifier["modifier_level"],
            "slug": modifier["slug"],
            "effect_recipe": modifier["effect_recipe"],
        }
        for modifier in modifiers
    ]
    with open(output_root / "effect_recipes.json", "w", encoding="utf-8") as handle:
        json.dump(recipe_summary, handle, indent=2)

    if failures:
        pd.DataFrame(failures, columns=["item", "reason"]).to_csv(output_root / "generation_failures.csv", index=False)

    print(f"ESC-50 root: {esc50_root}")
    print(f"Output root: {output_root}")
    print(f"Metadata: {metadata_path}")
    print(f"Source files: {len(meta)}")
    print(f"Triplets: {len(triplets)}")
    print(f"Failures: {len(failures)}")


# ============================================================
# 6. CLAP Embeddings
# ============================================================

def modifier_prompt_set(modifier):
    return [
        modifier,
        f"a {modifier} sound",
        f"a sound that is {modifier}",
        f"audio that sounds {modifier}",
        f"a recording with a {modifier} perceptual quality",
    ]


def l2_normalize(x):
    return F.normalize(x, dim=-1)


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_audio_for_clap(path, clap_sr=CLAP_SR):
    y, _ = librosa.load(str(path), sr=clap_sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y)

    max_len = clap_sr * 10
    if len(y) > max_len:
        start = (len(y) - max_len) // 2
        y = y[start:start + max_len]

    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 0:
        y = y / peak

    return y


@torch.inference_mode()
def embed_audio_paths(paths, dataset_root, save_path, model, tokenizer, feature_extractor, device, batch_size, recompute):
    if save_path.exists() and not recompute:
        return torch_load_any(save_path)

    all_embeddings = []

    for start in tqdm(range(0, len(paths), batch_size), desc=f"Embedding audio -> {save_path.name}"):
        batch_paths = paths[start:start + batch_size]
        audios = [load_audio_for_clap(dataset_root / path) for path in batch_paths]

        audio_inputs = feature_extractor(
            audios,
            sampling_rate=CLAP_SR,
            return_tensors="pt",
            padding=True,
            return_attention_mask=False,
        )
        text_inputs = tokenizer(
            ["a sound"] * len(audios),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {}
        inputs.update(text_inputs)
        inputs.update(audio_inputs)
        inputs = to_device(inputs, device)

        outputs = model(**inputs, return_dict=True)
        embeddings = l2_normalize(outputs.audio_embeds).detach().cpu()
        all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0).float()
    torch.save(all_embeddings, save_path)
    return all_embeddings


@torch.inference_mode()
def embed_modifier_texts(modifiers, save_path, model, tokenizer, feature_extractor, device, batch_size, recompute):
    if save_path.exists() and not recompute:
        return torch_load_any(save_path)

    modifier_embeddings = []
    prompt_sets = {}

    for modifier in tqdm(modifiers, desc="Embedding modifier texts"):
        prompts = modifier_prompt_set(modifier)
        prompt_sets[modifier] = prompts
        prompt_embeddings = []

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            text_inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            dummy_audios = [np.zeros(CLAP_SR, dtype=np.float32) for _ in batch]
            audio_inputs = feature_extractor(
                dummy_audios,
                sampling_rate=CLAP_SR,
                return_tensors="pt",
                padding=True,
                return_attention_mask=False,
            )

            inputs = {}
            inputs.update(text_inputs)
            inputs.update(audio_inputs)
            inputs = to_device(inputs, device)

            outputs = model(**inputs, return_dict=True)
            embeddings = l2_normalize(outputs.text_embeds).detach().cpu()
            prompt_embeddings.append(embeddings)

        prompt_embeddings = torch.cat(prompt_embeddings, dim=0)
        modifier_embedding = l2_normalize(prompt_embeddings.mean(dim=0, keepdim=True))[0]
        modifier_embeddings.append(modifier_embedding)

    payload = {
        "modifier_embeddings": torch.stack(modifier_embeddings, dim=0).float(),
        "modifier_texts": list(modifiers),
        "modifier_prompt_sets": prompt_sets,
    }
    torch.save(payload, save_path)
    return payload


def default_metadata_path(dataset_root, metadata_path=None):
    dataset_root = Path(dataset_root)
    if metadata_path is not None:
        return Path(metadata_path)
    preferred = [
        dataset_root / "metadata_triplets_11mod.csv",
        dataset_root / "metadata_triplets.csv",
    ]
    for path in preferred:
        if path.exists():
            return path
    raise FileNotFoundError("No metadata CSV was found.")


def default_index_path(dataset_root, metadata_path, index_path=None):
    if index_path is not None:
        return Path(index_path)
    name = "metadata_11mod_with_embedding_indices.csv" if "11mod" in Path(metadata_path).stem else "metadata_with_embedding_indices.csv"
    return Path(dataset_root) / name


def default_bundle_path(dataset_root, metadata_path, bundle_path=None):
    if bundle_path is not None:
        return Path(bundle_path)
    name = "clap_embeddings_11mod.pt" if "11mod" in Path(metadata_path).stem else "clap_embeddings.pt"
    return Path(dataset_root) / name


def precompute_embeddings(args):
    dataset_root = Path(args.dataset_root)
    metadata_path = default_metadata_path(dataset_root, args.metadata_path)
    index_path = default_index_path(dataset_root, metadata_path, args.index_path)
    bundle_path = default_bundle_path(dataset_root, metadata_path, args.bundle_path)

    df = pd.read_csv(metadata_path)
    source_paths = sorted(df["source_path"].unique().tolist())
    target_paths = sorted(df["target_path"].unique().tolist())
    modifier_texts = sorted(df["modifier"].unique().tolist())

    source_to_idx = {path: idx for idx, path in enumerate(source_paths)}
    target_to_idx = {path: idx for idx, path in enumerate(target_paths)}
    modifier_to_idx = {text: idx for idx, text in enumerate(modifier_texts)}

    indexed = df.copy()
    indexed["source_index"] = indexed["source_path"].map(source_to_idx)
    indexed["target_index"] = indexed["target_path"].map(target_to_idx)
    indexed["modifier_index"] = indexed["modifier"].map(modifier_to_idx)
    indexed.to_csv(index_path, index=False)

    device = choose_device(args.device)
    model = ClapModel.from_pretrained(args.model_id).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)

    prefix = "11mod_" if "11mod" in metadata_path.stem else ""
    source_embeddings = embed_audio_paths(
        source_paths,
        dataset_root,
        dataset_root / f"clap_{prefix}source_embeddings.pt",
        model,
        tokenizer,
        feature_extractor,
        device,
        args.audio_batch_size,
        args.recompute,
    )
    target_embeddings = embed_audio_paths(
        target_paths,
        dataset_root,
        dataset_root / f"clap_{prefix}target_embeddings.pt",
        model,
        tokenizer,
        feature_extractor,
        device,
        args.audio_batch_size,
        args.recompute,
    )
    modifier_payload = embed_modifier_texts(
        modifier_texts,
        dataset_root / f"clap_{prefix}modifier_embeddings.pt",
        model,
        tokenizer,
        feature_extractor,
        device,
        args.text_batch_size,
        args.recompute,
    )

    bundle = {
        "model_id": args.model_id,
        "clap_sample_rate": CLAP_SR,
        "dataset_root": str(dataset_root),
        "source_paths": source_paths,
        "target_paths": target_paths,
        "modifier_texts": modifier_texts,
        "modifier_prompt_sets": modifier_payload["modifier_prompt_sets"],
        "source_embeddings": source_embeddings,
        "target_embeddings": target_embeddings,
        "modifier_embeddings": modifier_payload["modifier_embeddings"],
    }
    torch.save(bundle, bundle_path)

    print(f"Indexed metadata: {index_path}")
    print(f"Embedding bundle: {bundle_path}")
    print(f"Source embeddings: {source_embeddings.shape}")
    print(f"Target embeddings: {target_embeddings.shape}")
    print(f"Modifier embeddings: {modifier_payload['modifier_embeddings'].shape}")


# ============================================================
# 7. Retrieval Metrics
# ============================================================

def metrics_from_ranks(ranks, ks=(1, 5, 10, 50)):
    ranks = np.asarray(ranks)
    metrics = {"queries": len(ranks)}
    for k in ks:
        metrics[f"R@{k}"] = float(np.mean(ranks <= k))
    metrics["MRR"] = float(np.mean(1.0 / ranks))
    metrics["median_rank"] = float(np.median(ranks))
    metrics["mean_rank"] = float(np.mean(ranks))
    return metrics


def build_query_embeddings(frame, source_bank, modifier_bank, method):
    source_indices = torch.tensor(frame["source_index"].values, dtype=torch.long, device=source_bank.device)
    modifier_indices = torch.tensor(frame["modifier_index"].values, dtype=torch.long, device=modifier_bank.device)
    source = source_bank[source_indices]
    modifier = modifier_bank[modifier_indices]

    if method == "audio_only":
        return F.normalize(source, dim=-1)
    if method == "text_only":
        return F.normalize(modifier, dim=-1)
    if method == "weighted_075_025":
        return F.normalize(0.75 * source + 0.25 * modifier, dim=-1)
    if method == "weighted_065_035":
        return F.normalize(0.65 * source + 0.35 * modifier, dim=-1)
    if method == "weighted_050_050":
        return F.normalize(0.50 * source + 0.50 * modifier, dim=-1)
    if method == "weighted_035_065":
        return F.normalize(0.35 * source + 0.65 * modifier, dim=-1)
    raise ValueError(f"Unknown method: {method}")


def compute_ranks(query_embeddings, correct_target_indices, candidate_target_indices, target_bank, batch_size, desc="Ranking"):
    device = query_embeddings.device
    candidate_tensor = torch.tensor(candidate_target_indices, dtype=torch.long, device=device)
    candidate_embeddings = F.normalize(target_bank[candidate_tensor], dim=-1)
    target_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(candidate_tensor.tolist())}
    correct_local = torch.tensor([target_to_local[int(idx)] for idx in correct_target_indices], dtype=torch.long, device=device)
    ranks = []

    for start in tqdm(range(0, len(query_embeddings), batch_size), desc=desc, leave=False):
        end = start + batch_size
        query = query_embeddings[start:end]
        correct_pos = correct_local[start:end]
        scores = query @ candidate_embeddings.T
        correct_scores = scores[torch.arange(scores.shape[0], device=device), correct_pos]
        batch_ranks = (scores > correct_scores[:, None]).sum(dim=1) + 1
        ranks.append(batch_ranks.detach().cpu())

    return torch.cat(ranks).numpy()


def evaluate_baselines(args):
    dataset_root = Path(args.dataset_root)
    metadata_path = default_metadata_path(dataset_root, args.metadata_path)
    index_path = default_index_path(dataset_root, metadata_path, args.index_path)
    bundle_path = default_bundle_path(dataset_root, metadata_path, args.bundle_path)
    output_dir = Path(args.output_dir) if args.output_dir is not None else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_path)
    bundle = torch_load_any(bundle_path)
    device = choose_device(args.device)

    source_bank = F.normalize(bundle["source_embeddings"].float(), dim=-1).to(device)
    target_bank = F.normalize(bundle["target_embeddings"].float(), dim=-1).to(device)
    modifier_bank = F.normalize(bundle["modifier_embeddings"].float(), dim=-1).to(device)

    methods = args.methods or [
        "audio_only",
        "text_only",
        "weighted_075_025",
        "weighted_065_035",
        "weighted_050_050",
        "weighted_035_065",
    ]

    summary_rows = []
    by_modifier_rows = []
    ranked_frames = []

    for split in args.splits:
        split_df = df[df["split"] == split].reset_index(drop=True)
        candidate_target_indices = sorted(split_df["target_index"].unique().tolist())
        correct_target_indices = split_df["target_index"].values

        for method in methods:
            query = build_query_embeddings(split_df, source_bank, modifier_bank, method)
            ranks = compute_ranks(
                query,
                correct_target_indices,
                candidate_target_indices,
                target_bank,
                args.batch_size,
                desc=f"{split} | {method}",
            )
            metrics = metrics_from_ranks(ranks)
            metrics.update({"split": split, "method": method})
            summary_rows.append(metrics)

            ranked = split_df.copy()
            ranked["method"] = method
            ranked["rank"] = ranks
            ranked_frames.append(ranked)

            for modifier, group in ranked.groupby("modifier"):
                row = metrics_from_ranks(group["rank"].values)
                row.update({
                    "split": split,
                    "method": method,
                    "modifier": modifier,
                    "modifier_level": group["modifier_level"].iloc[0],
                })
                by_modifier_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(output_dir / "zeroshot_baseline_summary.csv", index=False)
    pd.DataFrame(by_modifier_rows).to_csv(output_dir / "zeroshot_baseline_by_modifier.csv", index=False)
    pd.concat(ranked_frames, ignore_index=True).to_csv(output_dir / "zeroshot_baseline_ranks.csv", index=False)

    print(f"Baseline evaluation saved to: {output_dir}")


# ============================================================
# 8. Modifier Adapter
# ============================================================

class TripletEmbeddingDataset(Dataset):
    def __init__(self, frame):
        self.source_idx = torch.tensor(frame["source_index"].values, dtype=torch.long)
        self.modifier_idx = torch.tensor(frame["modifier_index"].values, dtype=torch.long)
        self.target_idx = torch.tensor(frame["target_index"].values, dtype=torch.long)

    def __len__(self):
        return len(self.source_idx)

    def __getitem__(self, idx):
        return self.source_idx[idx], self.modifier_idx[idx], self.target_idx[idx]


class ModifierAdapter(nn.Module):
    def __init__(self, emb_dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        input_dim = emb_dim * 4
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, source_emb, modifier_emb):
        x = torch.cat(
            [
                source_emb,
                modifier_emb,
                source_emb * modifier_emb,
                source_emb - modifier_emb,
            ],
            dim=-1,
        )
        h = self.norm(x)
        h = self.dropout(F.gelu(self.fc1(h)))
        h = self.dropout(F.gelu(self.fc2(h)))
        delta = self.fc3(h)
        return F.normalize(source_emb + delta, dim=-1)


class ModifierAdapterLayerNorm(nn.Module):
    def __init__(self, emb_dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim * 4, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_emb, modifier_emb):
        x = torch.cat(
            [
                source_emb,
                modifier_emb,
                source_emb * modifier_emb,
                source_emb - modifier_emb,
            ],
            dim=-1,
        )
        h = self.dropout(F.gelu(self.ln1(self.fc1(x))))
        h = self.dropout(F.gelu(self.ln2(self.fc2(h))))
        delta = self.fc3(h)
        return F.normalize(source_emb + delta, dim=-1)


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_adapter_checkpoint(path, emb_dim, device):
    checkpoint = torch_load_any(path)
    state = get_state_dict(checkpoint)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    hidden_dim = int(config.get("hidden_dim", state["fc1.weight"].shape[0]))
    dropout = float(config.get("dropout", 0.1))

    if "norm.weight" in state:
        model = ModifierAdapter(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif "ln1.weight" in state:
        model = ModifierAdapterLayerNorm(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError("Unsupported adapter checkpoint structure.")

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def contrastive_loss(query_emb, target_emb, temperature):
    logits = (query_emb @ target_emb.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_q_to_t = F.cross_entropy(logits, labels)
    loss_t_to_q = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_q_to_t + loss_t_to_q)


@torch.inference_mode()
def compute_adapter_ranks_for_frame(frame, adapter, source_bank, target_bank, modifier_bank, device, batch_size, method="modifier_adapter", desc="Evaluating"):
    frame = frame.reset_index(drop=True)
    candidate_target_indices = sorted(frame["target_index"].unique().tolist())
    candidate_tensor = torch.tensor(candidate_target_indices, dtype=torch.long, device=device)
    candidate_embeddings = F.normalize(target_bank[candidate_tensor], dim=-1)
    target_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(candidate_target_indices)}
    correct_local_all = torch.tensor(
        [target_to_local[int(idx)] for idx in frame["target_index"].values],
        dtype=torch.long,
        device=device,
    )
    loader = DataLoader(TripletEmbeddingDataset(frame), batch_size=batch_size, shuffle=False, num_workers=0)
    ranks = []
    cursor = 0
    adapter.eval()

    for source_idx, modifier_idx, target_idx in tqdm(loader, desc=desc, leave=False):
        batch_n = len(source_idx)
        source_idx = source_idx.to(device)
        modifier_idx = modifier_idx.to(device)
        source = source_bank[source_idx]
        modifier = modifier_bank[modifier_idx]

        if method == "modifier_adapter":
            query = adapter(source, modifier)
        elif method == "audio_only":
            query = F.normalize(source, dim=-1)
        elif method == "text_only":
            query = F.normalize(modifier, dim=-1)
        elif method == "weighted_065_035":
            query = F.normalize(0.65 * source + 0.35 * modifier, dim=-1)
        elif method == "weighted_050_050":
            query = F.normalize(0.50 * source + 0.50 * modifier, dim=-1)
        else:
            raise ValueError(f"Unknown method: {method}")

        scores = query @ candidate_embeddings.T
        correct_local = correct_local_all[cursor:cursor + batch_n]
        cursor += batch_n
        correct_scores = scores[torch.arange(batch_n, device=device), correct_local]
        batch_ranks = (scores > correct_scores[:, None]).sum(dim=1) + 1
        ranks.append(batch_ranks.detach().cpu())

    return torch.cat(ranks).numpy()


def train_adapter(args):
    dataset_root = Path(args.dataset_root)
    metadata_path = default_metadata_path(dataset_root, args.metadata_path)
    index_path = default_index_path(dataset_root, metadata_path, args.index_path)
    bundle_path = default_bundle_path(dataset_root, metadata_path, args.bundle_path)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path is not None else dataset_root / (
        "modifier_adapter_11mod_best.pt" if "11mod" in Path(index_path).stem else "modifier_adapter_best.pt"
    )
    train_log_path = Path(args.train_log_path) if args.train_log_path is not None else dataset_root / (
        "modifier_adapter_11mod_train_log.csv" if "11mod" in Path(index_path).stem else "modifier_adapter_train_log.csv"
    )
    eval_summary_path = dataset_root / (
        "modifier_adapter_11mod_eval_summary.csv" if "11mod" in Path(index_path).stem else "modifier_adapter_eval_summary.csv"
    )
    eval_by_modifier_path = dataset_root / (
        "modifier_adapter_11mod_eval_by_modifier.csv" if "11mod" in Path(index_path).stem else "modifier_adapter_eval_by_modifier.csv"
    )

    set_seed(args.seed)
    device = choose_device(args.device)
    bundle = torch_load_any(bundle_path)
    df = pd.read_csv(index_path)

    source_bank = F.normalize(bundle["source_embeddings"].float(), dim=-1).to(device)
    target_bank = F.normalize(bundle["target_embeddings"].float(), dim=-1).to(device)
    modifier_bank = F.normalize(bundle["modifier_embeddings"].float(), dim=-1).to(device)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_loader = DataLoader(
        TripletEmbeddingDataset(train_df),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    adapter = ModifierAdapter(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_r1 = -1.0
    best_val_mrr = -1.0
    best_epoch = -1
    bad_epochs = 0
    train_log = []

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total_loss = 0.0
        total_examples = 0

        for source_idx, modifier_idx, target_idx in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            source_idx = source_idx.to(device)
            modifier_idx = modifier_idx.to(device)
            target_idx = target_idx.to(device)
            source = source_bank[source_idx]
            modifier = modifier_bank[modifier_idx]
            target = target_bank[target_idx]

            query = adapter(source, modifier)
            loss = contrastive_loss(query, target, args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(source_idx)
            total_examples += len(source_idx)

        train_loss = total_loss / total_examples
        val_ranks = compute_adapter_ranks_for_frame(
            val_df,
            adapter,
            source_bank,
            target_bank,
            modifier_bank,
            device,
            args.eval_batch_size,
            method="modifier_adapter",
            desc=f"Val epoch {epoch}",
        )
        val_metrics = metrics_from_ranks(val_ranks)
        train_log.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{key}": value for key, value in val_metrics.items()}})

        val_r1 = val_metrics["R@1"]
        val_mrr = val_metrics["MRR"]
        improved = (val_r1 > best_val_r1) or (np.isclose(val_r1, best_val_r1) and val_mrr > best_val_mrr)

        if improved:
            best_val_r1 = val_r1
            best_val_mrr = val_mrr
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "state_dict": adapter.state_dict(),
                    "config": {
                        "emb_dim": args.emb_dim,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "temperature": args.temperature,
                        "source_model_id": bundle.get("model_id", "unknown"),
                        "modifier_set": sorted(df["modifier"].unique().tolist()),
                    },
                    "best_epoch": best_epoch,
                    "best_val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"val R@1={val_metrics['R@1']:.4f} | R@5={val_metrics['R@5']:.4f} | "
            f"R@10={val_metrics['R@10']:.4f} | MRR={val_metrics['MRR']:.4f} | best_epoch={best_epoch}"
        )

        if bad_epochs >= args.early_stop_patience:
            break

    pd.DataFrame(train_log).to_csv(train_log_path, index=False)
    adapter, checkpoint = load_adapter_checkpoint(checkpoint_path, args.emb_dim, device)
    methods = ["audio_only", "text_only", "weighted_065_035", "weighted_050_050", "modifier_adapter"]
    summary_rows = []
    by_modifier_rows = []

    for split_name, frame in [("val", val_df), ("test", test_df)]:
        for method in methods:
            ranks = compute_adapter_ranks_for_frame(
                frame,
                adapter,
                source_bank,
                target_bank,
                modifier_bank,
                device,
                args.eval_batch_size,
                method=method,
                desc=f"{split_name} | {method}",
            )
            metrics = metrics_from_ranks(ranks)
            metrics.update({"split": split_name, "method": method})
            summary_rows.append(metrics)

            ranked = frame.copy()
            ranked["rank"] = ranks
            for modifier, group in ranked.groupby("modifier"):
                row = metrics_from_ranks(group["rank"].values)
                row.update({
                    "split": split_name,
                    "method": method,
                    "modifier": modifier,
                    "modifier_level": group["modifier_level"].iloc[0],
                })
                by_modifier_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(eval_summary_path, index=False)
    pd.DataFrame(by_modifier_rows).to_csv(eval_by_modifier_path, index=False)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Train log: {train_log_path}")
    print(f"Evaluation summary: {eval_summary_path}")
    print(f"Evaluation by modifier: {eval_by_modifier_path}")


# ============================================================
# 9. Within-Source Evaluation
# ============================================================

@torch.inference_mode()
def evaluate_within_source_frame(frame, methods, adapter, source_bank, target_bank, modifier_bank, device, modifier_texts, split_name):
    frame = frame.reset_index(drop=True)
    predictions = []

    for source_path, group in tqdm(frame.groupby("source_path", sort=False), desc=f"Within-source eval: {split_name}"):
        group = group.reset_index(drop=True)
        candidate_target_indices = torch.tensor(group["target_index"].values, dtype=torch.long, device=device)
        candidate_embeddings = target_bank[candidate_target_indices]
        candidate_target_indices_cpu = candidate_target_indices.detach().cpu().numpy()
        candidate_modifiers = group["modifier"].astype(str).tolist()
        candidate_paths = group["target_path"].astype(str).tolist()

        for _, row in group.iterrows():
            source_idx = int(row["source_index"])
            modifier_idx = int(row["modifier_index"])
            correct_target_idx = int(row["target_index"])
            source = source_bank[source_idx].unsqueeze(0)
            modifier = modifier_bank[modifier_idx].unsqueeze(0)

            for method in methods:
                if method == "modifier_adapter":
                    query = adapter(source, modifier)
                elif method == "audio_only":
                    query = F.normalize(source, dim=-1)
                elif method == "text_only":
                    query = F.normalize(modifier, dim=-1)
                elif method == "weighted_065_035":
                    query = F.normalize(0.65 * source + 0.35 * modifier, dim=-1)
                elif method == "weighted_050_050":
                    query = F.normalize(0.50 * source + 0.50 * modifier, dim=-1)
                else:
                    raise ValueError(f"Unknown method: {method}")

                scores = (candidate_embeddings @ query.squeeze(0)).detach().cpu().numpy()
                order = np.argsort(-scores)
                ranked_target_indices = candidate_target_indices_cpu[order]
                correct_position = np.where(ranked_target_indices == correct_target_idx)[0]
                if len(correct_position) == 0:
                    raise RuntimeError("Correct target was not found among same-source candidates.")

                rank = int(correct_position[0]) + 1
                top_position = int(order[0])
                predictions.append({
                    "split": split_name,
                    "method": method,
                    "source_path": str(source_path),
                    "true_modifier": str(row["modifier"]),
                    "predicted_modifier": candidate_modifiers[top_position],
                    "target_path": str(row["target_path"]),
                    "predicted_target_path": candidate_paths[top_position],
                    "rank": rank,
                    "num_candidates": len(group),
                    "top1_correct": int(rank == 1),
                    "modifier_order": json.dumps(modifier_texts),
                })

    return pd.DataFrame(predictions)


def summarize_within_source(predictions):
    def metrics(group):
        ranks = group["rank"].to_numpy()
        row = metrics_from_ranks(ranks, ks=(1, 3, 5, 10))
        row["num_queries"] = row.pop("queries")
        return pd.Series(row)

    overall = predictions.groupby(["split", "method"], sort=False).apply(metrics).reset_index()
    by_modifier = predictions.groupby(["split", "method", "true_modifier"], sort=False).apply(metrics).reset_index()
    return overall, by_modifier


def save_confusion_matrix(predictions, modifier_texts, output_dir, split_name):
    adapter_predictions = predictions[
        (predictions["split"] == split_name)
        & (predictions["method"] == "modifier_adapter")
    ].copy()
    counts = pd.crosstab(
        adapter_predictions["true_modifier"],
        adapter_predictions["predicted_modifier"],
    ).reindex(index=modifier_texts, columns=modifier_texts, fill_value=0)
    normalized = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    counts_path = output_dir / f"adapter_confusion_counts_{split_name}.csv"
    normalized_path = output_dir / f"adapter_confusion_normalized_{split_name}.csv"
    figure_path = output_dir / f"adapter_confusion_matrix_{split_name}.png"

    counts.to_csv(counts_path)
    normalized.to_csv(normalized_path)

    plt.figure(figsize=(11, 8))
    plt.imshow(normalized.values, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(label="Prediction rate")
    plt.xticks(range(len(modifier_texts)), modifier_texts, rotation=45, ha="right")
    plt.yticks(range(len(modifier_texts)), modifier_texts)
    plt.xlabel("Predicted modifier")
    plt.ylabel("True modifier")
    plt.title(f"Modifier confusion matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def evaluate_within_source(args):
    dataset_root = Path(args.dataset_root)
    metadata_path = default_metadata_path(dataset_root, args.metadata_path)
    index_path = default_index_path(dataset_root, metadata_path, args.index_path)
    bundle_path = default_bundle_path(dataset_root, metadata_path, args.bundle_path)
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir) if args.output_dir is not None else dataset_root / (
        "within_source_11mod_results" if "11mod" in Path(index_path).stem else "within_source_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    df = pd.read_csv(index_path)
    bundle = torch_load_any(bundle_path)
    modifier_texts = list(bundle["modifier_texts"])

    source_bank = F.normalize(bundle["source_embeddings"].float(), dim=-1).to(device)
    target_bank = F.normalize(bundle["target_embeddings"].float(), dim=-1).to(device)
    modifier_bank = F.normalize(bundle["modifier_embeddings"].float(), dim=-1).to(device)
    adapter, checkpoint = load_adapter_checkpoint(checkpoint_path, source_bank.shape[1], device)

    methods = args.methods or ["audio_only", "text_only", "weighted_065_035", "weighted_050_050", "modifier_adapter"]
    parts = []

    for split_name in args.splits:
        frame = df[df["split"] == split_name].copy()
        if len(frame) > 0:
            parts.append(evaluate_within_source_frame(
                frame,
                methods,
                adapter,
                source_bank,
                target_bank,
                modifier_bank,
                device,
                modifier_texts,
                split_name,
            ))

    predictions = pd.concat(parts, ignore_index=True)
    overall, by_modifier = summarize_within_source(predictions)

    predictions.to_csv(output_dir / "within_source_predictions.csv", index=False)
    overall.to_csv(output_dir / "within_source_eval_summary.csv", index=False)
    by_modifier.to_csv(output_dir / "within_source_eval_by_modifier.csv", index=False)

    for split_name in args.splits:
        if split_name in predictions["split"].unique():
            save_confusion_matrix(predictions, modifier_texts, output_dir, split_name)

    print(f"Within-source results saved to: {output_dir}")


# ============================================================
# 10. Listening Evaluation Stimuli
# ============================================================

def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def copy_audio_normalized(src_path, dst_path, peak_db=-3.0, fade_ms=15):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(str(src_path), sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y)
    peak = np.max(np.abs(y)) if len(y) else 0.0
    target_peak = db_to_amp(peak_db)
    if peak > 0:
        y = y / peak * target_peak

    fade_len = int(sr * fade_ms / 1000)
    if fade_len > 0 and len(y) > 2 * fade_len:
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        y[:fade_len] *= fade_in
        y[-fade_len:] *= fade_out

    sf.write(dst_path, y, sr, subtype=OUTPUT_SUBTYPE)


@torch.inference_mode()
def retrieve_topk(row, candidate_df, adapter, source_bank, target_bank, modifier_bank, device, top_k):
    source = source_bank[int(row["source_index"])].unsqueeze(0)
    modifier = modifier_bank[int(row["modifier_index"])].unsqueeze(0)
    query = adapter(source, modifier)
    candidate_indices = torch.tensor(candidate_df["target_index"].values, dtype=torch.long, device=device)
    candidate_embeddings = target_bank[candidate_indices]
    scores = (query @ candidate_embeddings.T)[0]
    order = torch.argsort(scores, descending=True)[:top_k].detach().cpu().numpy()

    results = []
    for rank, pos in enumerate(order, start=1):
        candidate = candidate_df.iloc[int(pos)]
        results.append({
            "rank": rank,
            "score": float(scores[int(pos)].detach().cpu()),
            "target_path": candidate["target_path"],
            "modifier": candidate["modifier"],
            "source_path": candidate["source_path"],
            "esc50_category": candidate.get("esc50_category", ""),
            "target_index": int(candidate["target_index"]),
        })
    return results


def make_listening_stimuli(args):
    dataset_root = Path(args.dataset_root)
    metadata_path = default_metadata_path(dataset_root, args.metadata_path)
    index_path = default_index_path(dataset_root, metadata_path, args.index_path)
    bundle_path = default_bundle_path(dataset_root, metadata_path, args.bundle_path)
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    reference_dir = audio_dir / "references"
    candidate_dir = audio_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = choose_device(args.device)
    df = pd.read_csv(index_path)
    bundle = torch_load_any(bundle_path)
    source_bank = F.normalize(bundle["source_embeddings"].float(), dim=-1).to(device)
    target_bank = F.normalize(bundle["target_embeddings"].float(), dim=-1).to(device)
    modifier_bank = F.normalize(bundle["modifier_embeddings"].float(), dim=-1).to(device)
    adapter, checkpoint = load_adapter_checkpoint(checkpoint_path, source_bank.shape[1], device)

    test_df = df[df["split"] == args.split].copy().reset_index(drop=True)
    if args.exclude_categories:
        test_df = test_df[~test_df["esc50_category"].isin(args.exclude_categories)].reset_index(drop=True)

    modifiers = args.modifiers or sorted(test_df["modifier"].unique().tolist())
    test_df = test_df[test_df["modifier"].isin(modifiers)].reset_index(drop=True)
    candidate_pool = test_df.copy()

    source_blocks = (
        test_df.sort_values(["source_path", "modifier"])
        .drop_duplicates("source_path")
        .sample(frac=1, random_state=args.seed)
        .head(args.blocks)
        .reset_index(drop=True)
    )
    if len(source_blocks) < args.blocks:
        raise ValueError("Not enough source blocks were available for the requested stimuli set.")

    block_rows = []
    trial_rows = []
    topk_rows = []

    for block_idx, block in source_blocks.iterrows():
        block_id = block_idx + 1
        source_path = block["source_path"]
        reference_src = dataset_root / source_path
        category = block.get("esc50_category", "")
        reference_file = reference_dir / f"block_{block_id:02d}_reference_{slugify(str(category))}.wav"
        copy_audio_normalized(reference_src, reference_file)

        block_rows.append({
            "block_id": block_id,
            "source_path": source_path,
            "reference_file": str(reference_file.relative_to(output_dir)),
            "esc50_category": category,
        })

        for question_idx, modifier in enumerate(modifiers, start=1):
            query_rows = test_df[(test_df["source_path"] == source_path) & (test_df["modifier"] == modifier)]
            if len(query_rows) == 0:
                continue
            query_row = query_rows.iloc[0]
            topk = retrieve_topk(query_row, candidate_pool, adapter, source_bank, target_bank, modifier_bank, device, args.top_k)
            selected = topk[0]
            candidate_src = dataset_root / selected["target_path"]
            candidate_file = candidate_dir / f"block_{block_id:02d}_q_{question_idx:02d}_{slugify(modifier)}.wav"
            copy_audio_normalized(candidate_src, candidate_file)

            trial_id = f"B{block_id:02d}_Q{question_idx:02d}"
            trial_rows.append({
                "trial_id": trial_id,
                "block_id": block_id,
                "question_id": question_idx,
                "source_path": source_path,
                "reference_file": str(reference_file.relative_to(output_dir)),
                "modifier": modifier,
                "candidate_file": str(candidate_file.relative_to(output_dir)),
                "retrieved_target_path": selected["target_path"],
                "retrieved_rank": selected["rank"],
                "retrieved_score": selected["score"],
                "retrieved_modifier": selected["modifier"],
                "retrieved_source_path": selected["source_path"],
                "oracle_target_path": query_row["target_path"],
                "top1_is_oracle_target": bool(selected["target_path"] == query_row["target_path"]),
                "top1_same_source_as_query": bool(selected["source_path"] == source_path),
                "top1_same_modifier_as_query": bool(selected["modifier"] == modifier),
            })

            for result in topk:
                result = dict(result)
                result.update({
                    "trial_id": trial_id,
                    "block_id": block_id,
                    "question_id": question_idx,
                    "query_source_path": source_path,
                    "query_modifier": modifier,
                    "oracle_target_path": query_row["target_path"],
                })
                topk_rows.append(result)

    blocks = pd.DataFrame(block_rows)
    trials = pd.DataFrame(trial_rows)
    topk = pd.DataFrame(topk_rows)
    forms = trials[["trial_id", "block_id", "question_id", "reference_file", "modifier", "candidate_file"]].copy()
    forms["question_1"] = "Similarity to reference audio, 1-5"
    forms["question_2"] = "Match with text modifier, 1-5"
    forms["comment"] = "Optional comment"

    blocks.to_csv(output_dir / "listening_blocks.csv", index=False)
    trials.to_csv(output_dir / "listening_trials.csv", index=False)
    topk.to_csv(output_dir / "listening_topk_retrievals.csv", index=False)
    forms.to_csv(output_dir / "microsoft_forms_build_sheet.csv", index=False)

    instructions = "\n".join([
        "# Listening Evaluation Build Notes",
        "",
        f"This stimulus set contains {len(blocks)} reference-audio blocks and {len(modifiers)} modifier conditions per block.",
        "",
        "For each block:",
        "1. Add the reference audio first.",
        "2. Add the candidate audio files in the order shown in microsoft_forms_build_sheet.csv.",
        "3. Ask participants to rate similarity to the reference audio and match with the text modifier.",
        "4. Use a 1-5 rating scale.",
    ])
    (output_dir / "listening_evaluation_instructions.md").write_text(instructions, encoding="utf-8")

    html_lines = [
        "<html>",
        "<head><meta charset='utf-8'><title>Listening Evaluation Preview</title></head>",
        "<body>",
        "<h1>Listening Evaluation Stimuli Preview</h1>",
    ]
    for _, block in blocks.iterrows():
        html_lines.append(f"<hr><h2>Block {int(block['block_id']):02d} - {block['esc50_category']}</h2>")
        html_lines.append(f"<audio controls src='{block['reference_file']}'></audio>")
        subset = trials[trials["block_id"] == block["block_id"]].sort_values("question_id")
        for _, trial in subset.iterrows():
            html_lines.append(f"<h3>{trial['trial_id']} - {trial['modifier']}</h3>")
            html_lines.append(f"<audio controls src='{trial['candidate_file']}'></audio>")
    html_lines.extend(["</body>", "</html>"])
    (output_dir / "preview.html").write_text("\n".join(html_lines), encoding="utf-8")

    summary = {
        "n_blocks": int(len(blocks)),
        "n_modifiers": int(len(modifiers)),
        "n_trials": int(len(trials)),
        "selected_modifiers": modifiers,
        "random_seed": int(args.seed),
        "top1_oracle_rate": float(trials["top1_is_oracle_target"].mean()) if len(trials) else 0.0,
        "top1_same_source_rate": float(trials["top1_same_source_as_query"].mean()) if len(trials) else 0.0,
        "top1_same_modifier_rate": float(trials["top1_same_modifier_as_query"].mean()) if len(trials) else 0.0,
    }
    (output_dir / "stimuli_generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)

    print(f"Listening evaluation stimuli saved to: {output_dir}")


# ============================================================
# 11. Result Figures
# ============================================================

def load_first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return pd.read_csv(path)
    return None


def plot_results(args):
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir is not None else dataset_root / "paper_ready_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    zeroshot = load_first_existing([dataset_root / "zeroshot_baseline_summary.csv"])
    adapter = load_first_existing([
        dataset_root / "modifier_adapter_11mod_eval_summary.csv",
        dataset_root / "modifier_adapter_eval_summary.csv",
    ])
    adapter_by_modifier = load_first_existing([
        dataset_root / "modifier_adapter_11mod_eval_by_modifier.csv",
        dataset_root / "modifier_adapter_eval_by_modifier.csv",
    ])
    train_log = load_first_existing([
        dataset_root / "modifier_adapter_11mod_train_log.csv",
        dataset_root / "modifier_adapter_train_log.csv",
    ])

    if adapter is not None:
        overall = pd.concat([frame for frame in [zeroshot, adapter] if frame is not None], ignore_index=True, sort=False)
        overall.to_csv(output_dir / "table_overall_retrieval_comparison.csv", index=False)
        test_overall = overall[overall["split"] == "test"].copy()
        plot_methods = ["audio_only", "text_only", "weighted_065_035", "modifier_adapter"]
        labels = {
            "audio_only": "Audio-only",
            "text_only": "Text-only",
            "weighted_065_035": "Weighted CLAP",
            "modifier_adapter": "CLAP + Adapter",
        }
        plot_df = test_overall[test_overall["method"].isin(plot_methods)].copy()
        plot_df["method_label"] = plot_df["method"].map(labels)
        long = plot_df.melt(id_vars=["method_label"], value_vars=["R@1", "R@5", "R@10"], var_name="Metric", value_name="Score")
        pivot = long.pivot(index="Metric", columns="method_label", values="Score")
        pivot.plot(kind="bar", figsize=(8, 4.5))
        plt.ylim(0, 1.05)
        plt.xlabel("")
        plt.ylabel("Score")
        plt.title("Overall Test Retrieval Performance")
        plt.tight_layout()
        plt.savefig(output_dir / "fig_overall_test_retrieval_r_at_k.png", dpi=300)
        plt.close()

    if adapter_by_modifier is not None:
        test_by_modifier = adapter_by_modifier[
            (adapter_by_modifier["split"] == "test")
            & (adapter_by_modifier["method"] == "modifier_adapter")
        ].sort_values("R@1", ascending=True)
        test_by_modifier.to_csv(output_dir / "table_adapter_test_by_modifier.csv", index=False)
        plt.figure(figsize=(8, 5))
        plt.barh(test_by_modifier["modifier"], test_by_modifier["R@1"])
        plt.xlim(0, 1.05)
        plt.xlabel("R@1")
        plt.ylabel("")
        plt.title("Adapter Test R@1 by Modifier")
        plt.tight_layout()
        plt.savefig(output_dir / "fig_adapter_test_r1_by_modifier.png", dpi=300)
        plt.close()

    if train_log is not None and "epoch" in train_log.columns:
        plt.figure(figsize=(8, 4.5))
        if "val_R@1" in train_log.columns:
            plt.plot(train_log["epoch"], train_log["val_R@1"], label="Val R@1")
        if "val_MRR" in train_log.columns:
            plt.plot(train_log["epoch"], train_log["val_MRR"], label="Val MRR")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Modifier Adapter Validation Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "fig_adapter_training_curve.png", dpi=300)
        plt.close()

    print(f"Result figures saved to: {output_dir}")


# ============================================================
# 12. Command Line Interface
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Perceptual ESC-50 audio retrieval project pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-triplets")
    build.add_argument("--esc50-root")
    build.add_argument("--esc50-zip")
    build.add_argument("--extract-dir")
    build.add_argument("--output-root", required=True)
    build.add_argument("--modifier-set", choices=["9", "11"], default="11")
    build.add_argument("--metadata-name")
    build.add_argument("--max-source-files", type=int)
    build.add_argument("--copy-originals", action="store_true")
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(func=build_triplet_dataset)

    embed = subparsers.add_parser("embed")
    embed.add_argument("--dataset-root", required=True)
    embed.add_argument("--metadata-path")
    embed.add_argument("--index-path")
    embed.add_argument("--bundle-path")
    embed.add_argument("--model-id", default=MODEL_ID)
    embed.add_argument("--audio-batch-size", type=int, default=16)
    embed.add_argument("--text-batch-size", type=int, default=16)
    embed.add_argument("--device")
    embed.add_argument("--recompute", action="store_true")
    embed.set_defaults(func=precompute_embeddings)

    baseline = subparsers.add_parser("evaluate-baselines")
    baseline.add_argument("--dataset-root", required=True)
    baseline.add_argument("--metadata-path")
    baseline.add_argument("--index-path")
    baseline.add_argument("--bundle-path")
    baseline.add_argument("--output-dir")
    baseline.add_argument("--splits", nargs="+", default=["val", "test"])
    baseline.add_argument("--methods", nargs="+")
    baseline.add_argument("--batch-size", type=int, default=512)
    baseline.add_argument("--device")
    baseline.set_defaults(func=evaluate_baselines)

    train = subparsers.add_parser("train-adapter")
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--metadata-path")
    train.add_argument("--index-path")
    train.add_argument("--bundle-path")
    train.add_argument("--checkpoint-path")
    train.add_argument("--train-log-path")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--emb-dim", type=int, default=512)
    train.add_argument("--hidden-dim", type=int, default=1024)
    train.add_argument("--dropout", type=float, default=0.10)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--eval-batch-size", type=int, default=1024)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--temperature", type=float, default=0.07)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--early-stop-patience", type=int, default=12)
    train.add_argument("--device")
    train.set_defaults(func=train_adapter)

    within = subparsers.add_parser("evaluate-within-source")
    within.add_argument("--dataset-root", required=True)
    within.add_argument("--metadata-path")
    within.add_argument("--index-path")
    within.add_argument("--bundle-path")
    within.add_argument("--checkpoint-path", required=True)
    within.add_argument("--output-dir")
    within.add_argument("--splits", nargs="+", default=["val", "test"])
    within.add_argument("--methods", nargs="+")
    within.add_argument("--device")
    within.set_defaults(func=evaluate_within_source)

    listening = subparsers.add_parser("make-listening-stimuli")
    listening.add_argument("--dataset-root", required=True)
    listening.add_argument("--metadata-path")
    listening.add_argument("--index-path")
    listening.add_argument("--bundle-path")
    listening.add_argument("--checkpoint-path", required=True)
    listening.add_argument("--output-dir", required=True)
    listening.add_argument("--split", default="test")
    listening.add_argument("--blocks", type=int, default=8)
    listening.add_argument("--top-k", type=int, default=5)
    listening.add_argument("--seed", type=int, default=DEFAULT_SEED)
    listening.add_argument("--modifiers", nargs="+")
    listening.add_argument("--exclude-categories", nargs="+")
    listening.add_argument("--device")
    listening.set_defaults(func=make_listening_stimuli)

    figures = subparsers.add_parser("plot-results")
    figures.add_argument("--dataset-root", required=True)
    figures.add_argument("--output-dir")
    figures.set_defaults(func=plot_results)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
