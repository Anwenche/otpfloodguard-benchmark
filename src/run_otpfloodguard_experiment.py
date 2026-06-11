#!/usr/bin/env python3
"""Generate a reproducible OTP flooding benchmark and run lightweight baselines."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RANDOM_SEED = 42
EVALUATION_SEEDS = [3, 7, 13, 21, 42, 99, 123]
GENERATOR_SENSITIVITY_SEEDS = [11, 23, 42, 67, 101]
_LEGACY_SEED_TERM = "seed"
STALE_RESULT_FILES = [
    f"multi_{_LEGACY_SEED_TERM}_metrics.csv",
    f"multi_{_LEGACY_SEED_TERM}_summary.csv",
]


FEATURE_DESCRIPTIONS = {
    "window_seconds": "Aggregation window length in seconds.",
    "otp_requests": "Number of OTP requests observed in the window.",
    "request_velocity_per_sec": "OTP request count divided by the window length.",
    "unique_phone_count": "Number of distinct destination phone numbers in the window.",
    "unique_ip_count": "Number of distinct source IP addresses in the window.",
    "unique_device_count": "Number of distinct device identifiers in the window.",
    "ip_phone_ratio": "Unique IP count divided by unique phone count.",
    "device_phone_ratio": "Unique device count divided by unique phone count.",
    "prefix_concentration": "Share of requests concentrated in the most common phone prefix bucket.",
    "sequential_phone_score": "Normalized score indicating sequential or near-sequential phone-number patterns.",
    "country_entropy": "Normalized entropy of destination-country distribution.",
    "carrier_entropy": "Normalized entropy of carrier or network distribution.",
    "success_rate": "Share of OTP requests that lead to successful verification.",
    "failure_rate": "Share of OTP requests that fail verification.",
    "avg_interarrival_ms": "Average time between consecutive OTP requests in milliseconds.",
    "new_account_ratio": "Share of requests associated with newly created or first-seen accounts.",
    "night_request_ratio": "Share of requests occurring during local night-time hours.",
    "repeat_phone_ratio": "Share of repeated destination phone numbers in the window.",
    "repeat_ip_ratio": "Share of repeated source IP addresses in the window.",
    "risk_country_ratio": "Share of requests routed to destinations marked as higher risk.",
}


def clipped_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float) -> float:
    return float(np.clip(rng.normal(mean, sd), low, high))


def poisson(rng: np.random.Generator, lam: float, low: int, high: int) -> int:
    return int(np.clip(rng.poisson(lam), low, high))


def generate_sample(rng: np.random.Generator, attack_type: str) -> dict[str, float | int | str]:
    if attack_type == "normal":
        normal_burst = rng.random() < 0.18
        requests = poisson(rng, 18 if normal_burst else 4, 1, 55 if normal_burst else 18)
        unique_phones = int(np.clip(round(requests * clipped_normal(rng, 0.85, 0.12, 0.45, 1.0)), 1, requests))
        unique_ips = int(np.clip(round(requests * clipped_normal(rng, 0.55, 0.18, 0.15, 1.0)), 1, requests))
        unique_devices = int(np.clip(round(requests * clipped_normal(rng, 0.70, 0.15, 0.20, 1.0)), 1, requests))
        success_rate = clipped_normal(rng, 0.58 if normal_burst else 0.72, 0.18, 0.18, 1.0)
        prefix_concentration = clipped_normal(rng, 0.34 if normal_burst else 0.22, 0.11, 0.05, 0.65)
        sequential_score = clipped_normal(rng, 0.12 if normal_burst else 0.05, 0.08, 0.0, 0.35)
        country_entropy = clipped_normal(rng, 0.42 if normal_burst else 0.30, 0.20, 0.0, 0.90)
        carrier_entropy = clipped_normal(rng, 0.54 if normal_burst else 0.45, 0.20, 0.05, 0.98)
        avg_interarrival_ms = clipped_normal(rng, 2800 if normal_burst else 9500, 2100, 350, 25000)
        new_account_ratio = clipped_normal(rng, 0.52 if normal_burst else 0.32, 0.21, 0.0, 0.90)
        night_ratio = clipped_normal(rng, 0.28 if normal_burst else 0.18, 0.20, 0.0, 0.90)
    elif attack_type == "flooding":
        low_intensity = rng.random() < 0.16
        requests = poisson(rng, 26 if low_intensity else 62, 10 if low_intensity else 22, 140)
        unique_phones = int(np.clip(round(requests * clipped_normal(rng, 0.70, 0.16, 0.25, 1.0)), 5, requests))
        unique_ips = int(np.clip(round(requests * clipped_normal(rng, 0.10, 0.06, 0.02, 0.25)), 1, requests))
        unique_devices = int(np.clip(round(requests * clipped_normal(rng, 0.18, 0.08, 0.03, 0.35)), 1, requests))
        success_rate = clipped_normal(rng, 0.22 if low_intensity else 0.07, 0.10, 0.0, 0.46)
        prefix_concentration = clipped_normal(rng, 0.32 if low_intensity else 0.36, 0.15, 0.08, 0.75)
        sequential_score = clipped_normal(rng, 0.18, 0.13, 0.0, 0.62)
        country_entropy = clipped_normal(rng, 0.50 if low_intensity else 0.65, 0.20, 0.10, 1.0)
        carrier_entropy = clipped_normal(rng, 0.72, 0.16, 0.25, 1.0)
        avg_interarrival_ms = clipped_normal(rng, 1800 if low_intensity else 520, 900, 80, 4200)
        new_account_ratio = clipped_normal(rng, 0.64 if low_intensity else 0.78, 0.17, 0.22, 1.0)
        night_ratio = clipped_normal(rng, 0.42, 0.22, 0.0, 1.0)
    elif attack_type == "sms_pumping":
        low_intensity = rng.random() < 0.18
        requests = poisson(rng, 23 if low_intensity else 44, 8 if low_intensity else 15, 115)
        unique_phones = int(np.clip(round(requests * clipped_normal(rng, 0.92, 0.08, 0.55, 1.0)), 8, requests))
        unique_ips = int(np.clip(round(requests * clipped_normal(rng, 0.22, 0.11, 0.04, 0.48)), 1, requests))
        unique_devices = int(np.clip(round(requests * clipped_normal(rng, 0.28, 0.12, 0.05, 0.55)), 1, requests))
        success_rate = clipped_normal(rng, 0.18 if low_intensity else 0.04, 0.09, 0.0, 0.40)
        prefix_concentration = clipped_normal(rng, 0.48 if low_intensity else 0.58, 0.18, 0.14, 0.95)
        sequential_score = clipped_normal(rng, 0.26, 0.18, 0.0, 0.75)
        country_entropy = clipped_normal(rng, 0.42, 0.20, 0.02, 0.95)
        carrier_entropy = clipped_normal(rng, 0.22, 0.13, 0.0, 0.60)
        avg_interarrival_ms = clipped_normal(rng, 1900 if low_intensity else 760, 980, 100, 4500)
        new_account_ratio = clipped_normal(rng, 0.70 if low_intensity else 0.86, 0.14, 0.30, 1.0)
        night_ratio = clipped_normal(rng, 0.50, 0.24, 0.0, 1.0)
    elif attack_type == "sequential_spray":
        low_intensity = rng.random() < 0.18
        requests = poisson(rng, 20 if low_intensity else 35, 7 if low_intensity else 12, 95)
        unique_phones = int(np.clip(round(requests * clipped_normal(rng, 0.97, 0.04, 0.70, 1.0)), 8, requests))
        unique_ips = int(np.clip(round(requests * clipped_normal(rng, 0.18, 0.08, 0.03, 0.42)), 1, requests))
        unique_devices = int(np.clip(round(requests * clipped_normal(rng, 0.24, 0.10, 0.04, 0.50)), 1, requests))
        success_rate = clipped_normal(rng, 0.20 if low_intensity else 0.08, 0.09, 0.0, 0.42)
        prefix_concentration = clipped_normal(rng, 0.62 if low_intensity else 0.75, 0.16, 0.28, 0.98)
        sequential_score = clipped_normal(rng, 0.58 if low_intensity else 0.78, 0.18, 0.18, 1.0)
        country_entropy = clipped_normal(rng, 0.25, 0.16, 0.0, 0.75)
        carrier_entropy = clipped_normal(rng, 0.18, 0.12, 0.0, 0.55)
        avg_interarrival_ms = clipped_normal(rng, 2100 if low_intensity else 980, 1000, 120, 4800)
        new_account_ratio = clipped_normal(rng, 0.66 if low_intensity else 0.80, 0.17, 0.28, 1.0)
        night_ratio = clipped_normal(rng, 0.38, 0.22, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    failure_rate = 1.0 - success_rate
    ip_phone_ratio = unique_ips / max(unique_phones, 1)
    device_phone_ratio = unique_devices / max(unique_phones, 1)
    request_velocity = requests / 60.0
    repeat_phone_ratio = 1.0 - (unique_phones / max(requests, 1))
    repeat_ip_ratio = 1.0 - (unique_ips / max(requests, 1))
    risk_country_ratio = clipped_normal(rng, 0.08, 0.07, 0.0, 0.35) if attack_type == "normal" else clipped_normal(rng, 0.38, 0.20, 0.0, 1.0)

    return {
        "window_seconds": 60,
        "otp_requests": requests,
        "request_velocity_per_sec": request_velocity,
        "unique_phone_count": unique_phones,
        "unique_ip_count": unique_ips,
        "unique_device_count": unique_devices,
        "ip_phone_ratio": ip_phone_ratio,
        "device_phone_ratio": device_phone_ratio,
        "prefix_concentration": prefix_concentration,
        "sequential_phone_score": sequential_score,
        "country_entropy": country_entropy,
        "carrier_entropy": carrier_entropy,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "avg_interarrival_ms": avg_interarrival_ms,
        "new_account_ratio": new_account_ratio,
        "night_request_ratio": night_ratio,
        "repeat_phone_ratio": repeat_phone_ratio,
        "repeat_ip_ratio": repeat_ip_ratio,
        "risk_country_ratio": risk_country_ratio,
        "attack_type": attack_type,
        "label": 0 if attack_type == "normal" else 1,
    }


def build_dataset(n_samples: int = 12000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    attack_types = rng.choice(
        ["normal", "flooding", "sms_pumping", "sequential_spray"],
        size=n_samples,
        p=[0.58, 0.18, 0.16, 0.08],
    )
    rows = [generate_sample(rng, str(attack_type)) for attack_type in attack_types]
    df = pd.DataFrame(rows)
    return inject_hard_windows(df, rng)


def build_dataset_from_attack_counts(attack_type_counts: dict[str, int], seed: int) -> pd.DataFrame:
    """Generate a dataset with fixed attack-type counts and a different generator seed."""
    rng = np.random.default_rng(seed)
    attack_types = []
    for attack_type, count in attack_type_counts.items():
        attack_types.extend([attack_type] * int(count))
    attack_types = np.asarray(attack_types, dtype=object)
    rng.shuffle(attack_types)
    rows = [generate_sample(rng, str(attack_type)) for attack_type in attack_types]
    df = pd.DataFrame(rows)
    return inject_hard_windows(df, rng)


def refresh_derived_features(df: pd.DataFrame, idx: np.ndarray) -> None:
    df.loc[idx, "failure_rate"] = 1.0 - df.loc[idx, "success_rate"]
    df.loc[idx, "ip_phone_ratio"] = df.loc[idx, "unique_ip_count"] / df.loc[idx, "unique_phone_count"].clip(lower=1)
    df.loc[idx, "device_phone_ratio"] = df.loc[idx, "unique_device_count"] / df.loc[idx, "unique_phone_count"].clip(lower=1)
    df.loc[idx, "request_velocity_per_sec"] = df.loc[idx, "otp_requests"] / df.loc[idx, "window_seconds"]
    df.loc[idx, "repeat_phone_ratio"] = 1.0 - (df.loc[idx, "unique_phone_count"] / df.loc[idx, "otp_requests"].clip(lower=1))
    df.loc[idx, "repeat_ip_ratio"] = 1.0 - (df.loc[idx, "unique_ip_count"] / df.loc[idx, "otp_requests"].clip(lower=1))


def inject_hard_windows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add ambiguous but realistic windows to avoid an over-separable benchmark."""
    normal_idx = df.index[df["label"] == 0].to_numpy()
    attack_idx = df.index[df["label"] == 1].to_numpy()
    hard_normal = rng.choice(normal_idx, size=round(len(normal_idx) * 0.09), replace=False)
    hard_attack = rng.choice(attack_idx, size=round(len(attack_idx) * 0.11), replace=False)

    for idx in hard_normal:
        requests = int(rng.integers(18, 82))
        df.loc[idx, "otp_requests"] = requests
        df.loc[idx, "unique_phone_count"] = int(rng.integers(max(4, requests // 3), requests + 1))
        df.loc[idx, "unique_ip_count"] = int(rng.integers(1, max(2, requests // 4)))
        df.loc[idx, "unique_device_count"] = int(rng.integers(1, max(2, requests // 3)))
        df.loc[idx, "success_rate"] = clipped_normal(rng, 0.24, 0.13, 0.02, 0.55)
        df.loc[idx, "prefix_concentration"] = clipped_normal(rng, 0.52, 0.18, 0.12, 0.92)
        df.loc[idx, "sequential_phone_score"] = clipped_normal(rng, 0.34, 0.22, 0.0, 0.90)
        df.loc[idx, "country_entropy"] = clipped_normal(rng, 0.50, 0.22, 0.0, 1.0)
        df.loc[idx, "carrier_entropy"] = clipped_normal(rng, 0.46, 0.23, 0.0, 1.0)
        df.loc[idx, "avg_interarrival_ms"] = clipped_normal(rng, 1250, 850, 120, 4200)
        df.loc[idx, "new_account_ratio"] = clipped_normal(rng, 0.62, 0.20, 0.10, 1.0)
        df.loc[idx, "night_request_ratio"] = clipped_normal(rng, 0.42, 0.25, 0.0, 1.0)
        df.loc[idx, "risk_country_ratio"] = clipped_normal(rng, 0.30, 0.20, 0.0, 0.90)

    for idx in hard_attack:
        requests = int(rng.integers(8, 34))
        df.loc[idx, "otp_requests"] = requests
        df.loc[idx, "unique_phone_count"] = int(rng.integers(max(2, requests // 2), requests + 1))
        df.loc[idx, "unique_ip_count"] = int(rng.integers(max(1, requests // 5), max(2, requests)))
        df.loc[idx, "unique_device_count"] = int(rng.integers(max(1, requests // 4), max(2, requests)))
        df.loc[idx, "success_rate"] = clipped_normal(rng, 0.48, 0.18, 0.08, 0.85)
        df.loc[idx, "prefix_concentration"] = clipped_normal(rng, 0.34, 0.17, 0.08, 0.78)
        df.loc[idx, "sequential_phone_score"] = clipped_normal(rng, 0.18, 0.16, 0.0, 0.70)
        df.loc[idx, "country_entropy"] = clipped_normal(rng, 0.38, 0.20, 0.0, 0.90)
        df.loc[idx, "carrier_entropy"] = clipped_normal(rng, 0.42, 0.20, 0.0, 0.95)
        df.loc[idx, "avg_interarrival_ms"] = clipped_normal(rng, 3600, 2200, 350, 11000)
        df.loc[idx, "new_account_ratio"] = clipped_normal(rng, 0.46, 0.22, 0.0, 0.92)
        df.loc[idx, "night_request_ratio"] = clipped_normal(rng, 0.30, 0.22, 0.0, 1.0)
        df.loc[idx, "risk_country_ratio"] = clipped_normal(rng, 0.20, 0.16, 0.0, 0.75)

    refresh_derived_features(df, hard_normal)
    refresh_derived_features(df, hard_attack)
    return df


def make_window_variant(df: pd.DataFrame, window_seconds: int, seed: int) -> pd.DataFrame:
    """Approximate a shorter or longer detection window from the base 60-second benchmark."""
    rng = np.random.default_rng(seed + window_seconds)
    variant = df.copy()
    scale = window_seconds / 60.0
    count_cols = ["otp_requests", "unique_phone_count", "unique_ip_count", "unique_device_count"]

    for col in count_cols:
        jitter = rng.normal(1.0, 0.06, size=len(variant))
        variant[col] = np.maximum(1, np.round(variant[col] * scale * jitter)).astype(int)

    variant["unique_phone_count"] = np.minimum(variant["unique_phone_count"], variant["otp_requests"])
    variant["unique_ip_count"] = np.minimum(variant["unique_ip_count"], variant["otp_requests"])
    variant["unique_device_count"] = np.minimum(variant["unique_device_count"], variant["otp_requests"])
    variant["avg_interarrival_ms"] = np.maximum(
        80,
        variant["avg_interarrival_ms"] / scale * rng.normal(1.0, 0.04, size=len(variant)),
    )
    variant["window_seconds"] = window_seconds
    refresh_derived_features(variant, variant.index.to_numpy())
    return variant


def make_difficulty_variant(df: pd.DataFrame, difficulty: str, seed: int) -> pd.DataFrame:
    """Create easy, overlap, or adaptive benchmark variants from the base benchmark."""
    rng = np.random.default_rng(seed + {"easy": 101, "overlap": 202, "adaptive": 303}[difficulty])
    variant = df.copy()
    attack_idx = variant.index[variant["label"] == 1].to_numpy()
    normal_idx = variant.index[variant["label"] == 0].to_numpy()

    if difficulty == "easy":
        variant.loc[attack_idx, "otp_requests"] = np.round(variant.loc[attack_idx, "otp_requests"] * 1.35).clip(1, 160).astype(int)
        variant.loc[attack_idx, "success_rate"] = (variant.loc[attack_idx, "success_rate"] * 0.55).clip(0, 1)
        variant.loc[attack_idx, "prefix_concentration"] = (variant.loc[attack_idx, "prefix_concentration"] * 1.18).clip(0, 1)
        variant.loc[attack_idx, "repeat_ip_ratio"] = (variant.loc[attack_idx, "repeat_ip_ratio"] * 1.12).clip(0, 1)
        variant.loc[normal_idx, "success_rate"] = (variant.loc[normal_idx, "success_rate"] * 1.08).clip(0, 1)
        variant.loc[normal_idx, "prefix_concentration"] = (variant.loc[normal_idx, "prefix_concentration"] * 0.85).clip(0, 1)
    elif difficulty == "adaptive":
        blend_features = [c for c in FEATURE_DESCRIPTIONS if c != "window_seconds"]
        variant[blend_features] = variant[blend_features].astype(float)
        sampled_normal = variant.loc[
            rng.choice(normal_idx, size=len(attack_idx), replace=True),
            blend_features,
        ].to_numpy()
        for feature in blend_features:
            col_pos = blend_features.index(feature)
            noise = rng.normal(1.0, 0.05, size=len(attack_idx))
            variant.loc[attack_idx, feature] = (
                0.22 * variant.loc[attack_idx, feature].to_numpy()
                + 0.78 * sampled_normal[:, col_pos] * noise
            )
    elif difficulty != "overlap":
        raise ValueError(f"Unknown difficulty: {difficulty}")

    for count_col in ["otp_requests", "unique_phone_count", "unique_ip_count", "unique_device_count"]:
        variant[count_col] = np.round(variant[count_col]).clip(1, None).astype(int)
    variant["unique_phone_count"] = np.minimum(variant["unique_phone_count"], variant["otp_requests"])
    variant["unique_ip_count"] = np.minimum(variant["unique_ip_count"], variant["otp_requests"])
    variant["unique_device_count"] = np.minimum(variant["unique_device_count"], variant["otp_requests"])
    variant["difficulty"] = difficulty
    refresh_derived_features(variant, variant.index.to_numpy())
    return variant


def make_attack_intensity_variant(df: pd.DataFrame, intensity: float, seed: int) -> pd.DataFrame:
    """Move attack windows toward or away from normal means to test attack intensity."""
    rng = np.random.default_rng(seed + int(intensity * 1000))
    variant = df.copy()
    attack_idx = variant.index[variant["label"] == 1].to_numpy()
    normal_idx = variant.index[variant["label"] == 0].to_numpy()
    attack_signal_features = [c for c in FEATURE_DESCRIPTIONS if c != "window_seconds"]
    variant[attack_signal_features] = variant[attack_signal_features].astype(float)
    sampled_normal = variant.loc[
        rng.choice(normal_idx, size=len(attack_idx), replace=True),
        attack_signal_features,
    ].to_numpy()
    for feature in attack_signal_features:
        col_pos = attack_signal_features.index(feature)
        attack_values = variant.loc[attack_idx, feature].to_numpy()
        jitter = rng.normal(1.0, 0.04, size=len(attack_idx))
        variant.loc[attack_idx, feature] = (
            sampled_normal[:, col_pos] + intensity * (attack_values - sampled_normal[:, col_pos])
        ) * jitter
    for count_col in ["otp_requests", "unique_phone_count", "unique_ip_count", "unique_device_count"]:
        variant[count_col] = np.round(variant[count_col]).clip(1, None).astype(int)
    variant["unique_phone_count"] = np.minimum(variant["unique_phone_count"], variant["otp_requests"])
    variant["unique_ip_count"] = np.minimum(variant["unique_ip_count"], variant["otp_requests"])
    variant["unique_device_count"] = np.minimum(variant["unique_device_count"], variant["otp_requests"])
    bounded_cols = [
        "success_rate",
        "failure_rate",
        "prefix_concentration",
        "sequential_phone_score",
        "country_entropy",
        "carrier_entropy",
        "new_account_ratio",
        "night_request_ratio",
        "repeat_phone_ratio",
        "repeat_ip_ratio",
        "risk_country_ratio",
    ]
    for col in bounded_cols:
        variant.loc[attack_idx, col] = variant.loc[attack_idx, col].clip(0, 1)
    variant.loc[attack_idx, "success_rate"] = 1.0 - variant.loc[attack_idx, "failure_rate"]
    refresh_derived_features(variant, variant.index.to_numpy())
    return variant


def make_generator_shift_variant(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create an out-of-generator test variant with shifted simulation assumptions."""
    rng = np.random.default_rng(seed + 707)
    variant = df.copy()
    normal_idx = variant.index[variant["label"] == 0].to_numpy()
    attack_idx = variant.index[variant["label"] == 1].to_numpy()

    # Alternative benign generator: more legitimate bursts and delivery failures.
    normal_count_noise = rng.normal(1.18, 0.12, size=len(normal_idx))
    variant.loc[normal_idx, "otp_requests"] = np.round(
        variant.loc[normal_idx, "otp_requests"] * normal_count_noise
    ).clip(1, 120).astype(int)
    variant.loc[normal_idx, "success_rate"] = (
        variant.loc[normal_idx, "success_rate"].to_numpy()
        - rng.normal(0.07, 0.05, size=len(normal_idx))
    ).clip(0.05, 1.0)
    variant.loc[normal_idx, "prefix_concentration"] = (
        variant.loc[normal_idx, "prefix_concentration"] * rng.normal(1.16, 0.10, size=len(normal_idx))
    ).clip(0.0, 1.0)
    variant.loc[normal_idx, "repeat_ip_ratio"] = (
        variant.loc[normal_idx, "repeat_ip_ratio"] * rng.normal(1.10, 0.10, size=len(normal_idx))
    ).clip(0.0, 1.0)
    variant.loc[normal_idx, "avg_interarrival_ms"] = (
        variant.loc[normal_idx, "avg_interarrival_ms"] * rng.normal(0.84, 0.12, size=len(normal_idx))
    ).clip(80, 25000)

    # Alternative attacker generator: lower rate, more distributed infrastructure, higher apparent success.
    attack_count_noise = rng.normal(0.78, 0.12, size=len(attack_idx))
    variant.loc[attack_idx, "otp_requests"] = np.round(
        variant.loc[attack_idx, "otp_requests"] * attack_count_noise
    ).clip(1, 140).astype(int)
    variant.loc[attack_idx, "success_rate"] = (
        variant.loc[attack_idx, "success_rate"].to_numpy()
        + rng.normal(0.14, 0.07, size=len(attack_idx))
    ).clip(0.0, 0.95)
    for col, mean, sd in [
        ("prefix_concentration", 0.82, 0.12),
        ("sequential_phone_score", 0.82, 0.12),
        ("repeat_ip_ratio", 0.76, 0.12),
        ("risk_country_ratio", 0.84, 0.14),
        ("new_account_ratio", 0.90, 0.10),
    ]:
        variant.loc[attack_idx, col] = (
            variant.loc[attack_idx, col] * rng.normal(mean, sd, size=len(attack_idx))
        ).clip(0.0, 1.0)
    variant.loc[attack_idx, "avg_interarrival_ms"] = (
        variant.loc[attack_idx, "avg_interarrival_ms"] * rng.normal(1.25, 0.18, size=len(attack_idx))
    ).clip(80, 25000)

    for count_col in ["unique_phone_count", "unique_ip_count", "unique_device_count"]:
        jitter = rng.normal(1.08, 0.12, size=len(attack_idx))
        variant.loc[attack_idx, count_col] = np.round(
            variant.loc[attack_idx, count_col] * jitter
        ).clip(1, None).astype(int)
    for count_col in ["unique_phone_count", "unique_ip_count", "unique_device_count"]:
        variant[count_col] = np.minimum(variant[count_col], variant["otp_requests"]).clip(1).astype(int)

    variant["generator_variant"] = "shifted_v2"
    refresh_derived_features(variant, variant.index.to_numpy())
    return variant


def prediction_metrics(
    model_name: str,
    feature_set: str,
    num_features: int,
    y_true,
    y_pred,
    y_score=None,
) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "model": model_name,
        "feature_set": feature_set,
        "num_features": num_features,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        row["roc_auc"] = roc_auc_score(y_true, y_score)
        row["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        row["roc_auc"] = float("nan")
        row["pr_auc"] = float("nan")
    return {
        **row,
    }


def model_scores(model, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores)
    return None


def evaluate_model(model_name: str, model, X_train, X_test, y_train, y_test, features: list[str]) -> dict[str, float | str | int]:
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])
    y_score = model_scores(model, X_test[features])
    feature_set = f"top_{len(features)}" if len(features) < X_train.shape[1] else "full"
    return prediction_metrics(model_name, feature_set, len(features), y_test, y_pred, y_score)


def write_environment_config() -> None:
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (RESULTS_DIR / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


def write_model_config(models: dict[str, object], feature_count: int, probability_threshold: float = 0.5) -> None:
    config = {
        "metadata": {
            "feature_count": feature_count,
            "default_model_score_threshold": probability_threshold,
            "score_interpretation": "model scores are empirical decision scores; probability calibration is not evaluated",
            "selection_scope": "all configurations fixed before held-out test evaluation",
            "standard_scaler": "used inside Logistic Regression pipeline and fitted on training folds/splits only",
        },
        "estimators": {model_name: model.get_params(deep=True) for model_name, model in models.items()},
    }
    (RESULTS_DIR / "model_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def write_feature_ranking_metadata(ranking_features: list[str]) -> None:
    metadata = {
        "ranking_method": "Random Forest impurity-based feature_importances_",
        "ranking_estimator": {
            "model": "RandomForestClassifier",
            "n_estimators": 250,
            "max_depth": 10,
            "min_samples_leaf": 3,
            "random_state": RANDOM_SEED,
        },
        "training_only": True,
        "held_out_test_used": False,
        "used_for_top_k_selection": True,
        "top_k_sets": [5, 10, 15],
        "permutation_importance_used_for_selection": False,
        "permutation_importance_role": "separate interpretability diagnostic computed on training-fold validation data",
        "bias_note": "impurity-based importance is used as a ranking heuristic and may favor features with more split opportunities",
        "ranked_features": ranking_features,
    }
    (RESULTS_DIR / "feature_ranking_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def write_reporting_precision_config() -> None:
    config = {
        "reported_decimal_places": 4,
        "mean_std_decimal_places": 4,
        "rounding": "Python format specification .4f",
        "note": "CSV files retain full floating-point precision; manuscript values are rounded to four decimals.",
    }
    (RESULTS_DIR / "reporting_precision.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_rule_config(rule: dict[str, float], velocity_failure_rule: dict[str, float]) -> None:
    config = {
        "tuned_four_signal_rule": {
            "selected_parameters": rule,
            "candidate_grid": {
                "request_threshold": [10, 15, 20, 25, 30, 40, 50],
                "failure_threshold": [0.30, 0.40, 0.50, 0.60, 0.70],
                "prefix_threshold": [0.35, 0.45, 0.55, 0.65],
                "repeat_ip_threshold": [0.45, 0.60, 0.75, 0.85],
            },
            "selection_metric": "mean_validation_f1",
            "selection_scope": "five_fold_stratified_cv_within_training_split",
        },
        "velocity_failure_rule": {
            "selected_parameters": velocity_failure_rule,
            "candidate_grid": {
                "request_threshold": [10, 15, 20, 25, 30, 40, 50],
                "failure_threshold": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            },
            "selection_metric": "mean_validation_f1",
            "selection_scope": "five_fold_stratified_cv_within_training_split",
        },
    }
    (RESULTS_DIR / "rule_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _mean_cv_f1_for_rule(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rule: dict[str, float],
    apply_fn,
) -> float:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    y_array = y_train.to_numpy()
    for _, val_pos in splitter.split(X_train, y_array):
        val_index = X_train.index[val_pos]
        pred = apply_fn(X_train.loc[val_index], rule)
        scores.append(f1_score(y_train.loc[val_index], pred, zero_division=0))
    return float(np.mean(scores))


def fit_rule_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    best_rule = {"cv_f1": -1.0}
    for request_threshold in [10, 15, 20, 25, 30, 40, 50]:
        for failure_threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
            for prefix_threshold in [0.35, 0.45, 0.55, 0.65]:
                for repeat_ip_threshold in [0.45, 0.60, 0.75, 0.85]:
                    candidate = {
                        "request_threshold": request_threshold,
                        "failure_threshold": failure_threshold,
                        "prefix_threshold": prefix_threshold,
                        "repeat_ip_threshold": repeat_ip_threshold,
                    }
                    score = _mean_cv_f1_for_rule(X_train, y_train, candidate, apply_rule_baseline)
                    if score > best_rule["cv_f1"]:
                        best_rule = {
                            **candidate,
                            "cv_f1": score,
                            "selection_method": "five_fold_stratified_cv_on_training_split",
                            "rule_logic": "(otp_requests >= request_threshold and failure_rate >= failure_threshold) or (prefix_concentration >= prefix_threshold and repeat_ip_ratio >= repeat_ip_threshold)",
                        }
    return best_rule


def fit_velocity_failure_rule(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    best_rule = {"cv_f1": -1.0}
    for request_threshold in [10, 15, 20, 25, 30, 40, 50]:
        for failure_threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            candidate = {
                "request_threshold": request_threshold,
                "failure_threshold": failure_threshold,
            }
            score = _mean_cv_f1_for_rule(X_train, y_train, candidate, apply_velocity_failure_rule)
            if score > best_rule["cv_f1"]:
                best_rule = {
                    **candidate,
                    "cv_f1": score,
                    "selection_method": "five_fold_stratified_cv_on_training_split",
                    "rule_logic": "otp_requests >= request_threshold and failure_rate >= failure_threshold",
                }
    return best_rule


def apply_rule_baseline(X: pd.DataFrame, rule: dict[str, float]) -> np.ndarray:
    velocity_condition = X["otp_requests"] >= rule["request_threshold"]
    failure_condition = X["failure_rate"] >= rule["failure_threshold"]
    prefix_condition = X["prefix_concentration"] >= rule["prefix_threshold"]
    repeat_condition = X["repeat_ip_ratio"] >= rule["repeat_ip_threshold"]
    return ((velocity_condition & failure_condition) | (prefix_condition & repeat_condition)).astype(int).to_numpy()


def apply_velocity_failure_rule(X: pd.DataFrame, rule: dict[str, float]) -> np.ndarray:
    velocity_condition = X["otp_requests"] >= rule["request_threshold"]
    failure_condition = X["failure_rate"] >= rule["failure_threshold"]
    return (velocity_condition & failure_condition).astype(int).to_numpy()


def write_feature_dictionary(df: pd.DataFrame, feature_cols: list[str]) -> None:
    rows = []
    for feature in feature_cols:
        rows.append(
            {
                "feature": feature,
                "definition": FEATURE_DESCRIPTIONS.get(feature, ""),
                "min": df[feature].min(),
                "max": df[feature].max(),
                "mean": df[feature].mean(),
                "std": df[feature].std(),
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "feature_dictionary.csv", index=False)


def write_benchmark_config(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().to_dict()
    total = int(len(df))
    normal_windows = int(counts.get(0, 0))
    attack_windows = int(counts.get(1, 0))
    config = {
        "total_windows": total,
        "normal_windows": normal_windows,
        "attack_windows": attack_windows,
        "attack_prevalence": attack_windows / total,
        "interpretation": "controlled benchmark prevalence, not production base rate",
    }
    (RESULTS_DIR / "benchmark_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_multi_split_evaluation(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for seed in EVALUATION_SEEDS:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=0.2,
            random_state=seed,
            stratify=df["label"],
        )
        X_train = df.loc[train_idx, feature_cols]
        X_test = df.loc[test_idx, feature_cols]
        y_train = df.loc[train_idx, "label"]
        y_test = df.loc[test_idx, "label"]
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                ]
            ),
        }
        for model_name, model in models.items():
            result = evaluate_model(model_name, model, X_train, X_test, y_train, y_test, feature_cols)
            result["split_seed"] = seed
            rows.append(result)

        rule = fit_rule_baseline(X_train, y_train)
        rule_pred = apply_rule_baseline(X_test, rule)
        rule_result = prediction_metrics("Tuned Rule Baseline", "domain_rules", 4, y_test, rule_pred)
        rule_result["split_seed"] = seed
        rows.append(rule_result)

        vf_rule = fit_velocity_failure_rule(X_train, y_train)
        vf_pred = apply_velocity_failure_rule(X_test, vf_rule)
        vf_result = prediction_metrics("Velocity+Failure Rule", "velocity_failure_rules", 2, y_test, vf_pred)
        vf_result["split_seed"] = seed
        rows.append(vf_result)
    seed_metrics = pd.DataFrame(rows)
    seed_metrics.to_csv(RESULTS_DIR / "multi_split_metrics.csv", index=False)
    summary = seed_metrics.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
    ).reset_index().sort_values("model")
    summary.to_csv(RESULTS_DIR / "multi_split_summary.csv", index=False)
    return seed_metrics


def evaluate_baselines_on_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_seed: int,
    generator_seed: int,
) -> list[dict[str, float | str | int]]:
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        random_state=split_seed,
        stratify=df["label"],
    )
    X_train = df.loc[train_idx, feature_cols]
    X_test = df.loc[test_idx, feature_cols]
    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        ),
    }
    rows: list[dict[str, float | str | int]] = []
    for model_name, model in models.items():
        result = evaluate_model(model_name, model, X_train, X_test, y_train, y_test, feature_cols)
        rows.append(result)

    rule = fit_rule_baseline(X_train, y_train)
    rule_pred = apply_rule_baseline(X_test, rule)
    rows.append(prediction_metrics("Tuned Rule Baseline", "domain_rules", 4, y_test, rule_pred))

    vf_rule = fit_velocity_failure_rule(X_train, y_train)
    vf_pred = apply_velocity_failure_rule(X_test, vf_rule)
    rows.append(prediction_metrics("Velocity+Failure Rule", "velocity_failure_rules", 2, y_test, vf_pred))

    n_samples = int(len(df))
    n_attack = int((df["label"] == 1).sum())
    n_normal = int((df["label"] == 0).sum())
    for row in rows:
        row["generator_seed"] = generator_seed
        row["split_seed"] = split_seed
        row["n_samples"] = n_samples
        row["n_normal"] = n_normal
        row["n_attack"] = n_attack
    return rows


def run_generator_seed_sensitivity(
    attack_type_counts: dict[str, int],
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for generator_seed in GENERATOR_SENSITIVITY_SEEDS:
        variant = build_dataset_from_attack_counts(attack_type_counts, generator_seed)
        rows.extend(
            evaluate_baselines_on_dataset(
                variant,
                feature_cols,
                split_seed=RANDOM_SEED,
                generator_seed=generator_seed,
            )
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "generator_seed_metrics.csv", index=False)
    summary = (
        metrics.groupby("model")
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            n_generator_seeds=("generator_seed", "nunique"),
        )
        .reset_index()
        .sort_values("model")
    )
    summary.to_csv(RESULTS_DIR / "generator_seed_summary.csv", index=False)
    config = {
        "generator_seeds": GENERATOR_SENSITIVITY_SEEDS,
        "fixed_split_seed": RANDOM_SEED,
        "difficulty": "Overlap",
        "models": [
            "Random Forest",
            "Gradient Boosting",
            "Logistic Regression",
            "Tuned Rule Baseline",
            "Velocity + Failure Rule",
        ],
        "fixed_attack_type_counts": {str(k): int(v) for k, v in attack_type_counts.items()},
        "reruns_full_diagnostics": False,
        "purpose": "measure sensitivity to generator randomness",
        "selection_boundaries": "rules, scalers, models, and feature handling are fitted within each generated dataset training portion only",
    }
    (RESULTS_DIR / "generator_seed_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return metrics


def run_ablation_study(X_train, X_test, y_train, y_test, feature_cols: list[str]) -> pd.DataFrame:
    ablations = {
        "full": [],
        "remove_success_failure": ["success_rate", "failure_rate"],
        "remove_velocity": ["otp_requests", "request_velocity_per_sec", "avg_interarrival_ms"],
        "remove_prefix_sequence": ["prefix_concentration", "sequential_phone_score"],
        "remove_ip_device_reuse": ["unique_ip_count", "unique_device_count", "ip_phone_ratio", "device_phone_ratio", "repeat_ip_ratio"],
        "remove_context": ["country_entropy", "carrier_entropy", "risk_country_ratio", "night_request_ratio", "new_account_ratio"],
    }
    rows = []
    for name, removed in ablations.items():
        features = [f for f in feature_cols if f not in removed]
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        result = evaluate_model("Random Forest", model, X_train, X_test, y_train, y_test, features)
        result["feature_set"] = name
        result["removed_features"] = ",".join(removed)
        rows.append(result)
    ablation = pd.DataFrame(rows).sort_values("f1", ascending=False)
    ablation.to_csv(RESULTS_DIR / "ablation_metrics.csv", index=False)
    return ablation


def select_representative_error_case(analyzed: pd.DataFrame, error_type: str, feature_cols: list[str]) -> tuple[int | None, float | None]:
    group = analyzed[analyzed["error_type"] == error_type]
    if group.empty:
        return None, None
    feature_matrix = group[feature_cols].astype(float)
    reference = feature_matrix.median()
    scale = analyzed[feature_cols].astype(float).std().replace(0, 1.0).fillna(1.0)
    distances = (((feature_matrix - reference) / scale) ** 2).sum(axis=1) ** 0.5
    ordered = pd.DataFrame({"row_id": group.index, "distance": distances.to_numpy()}).sort_values(
        ["distance", "row_id"],
        ascending=[True, True],
    )
    selected = ordered.iloc[0]
    return int(selected["row_id"]), float(selected["distance"])


def run_error_analysis(test_df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, feature_cols: list[str]) -> None:
    analyzed = test_df.copy()
    analyzed["predicted_label"] = y_pred
    analyzed["error_type"] = np.select(
        [
            (y_true == 0) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
            (y_true == 1) & (y_pred == 1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="unknown",
    )
    analyzed.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    summary = (
        analyzed.groupby("error_type")[feature_cols + ["label"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.insert(1, "count", analyzed.groupby("error_type").size().reindex(summary["error_type"]).to_numpy())
    summary.to_csv(RESULTS_DIR / "error_analysis_summary.csv", index=False)
    error_mix = (
        analyzed.groupby(["error_type", "attack_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["error_type", "count"], ascending=[True, False])
    )
    error_mix.to_csv(RESULTS_DIR / "error_type_attack_mix.csv", index=False)

    fp_row_id, fp_distance = select_representative_error_case(analyzed, "FP", feature_cols)
    fn_row_id, fn_distance = select_representative_error_case(analyzed, "FN", feature_cols)
    selection = {
        "selection_method": "closest_to_error_group_median",
        "representative_meaning": "closest to median feature profile, not highest severity or highest operational cost",
        "distance_metric": "euclidean",
        "feature_space": feature_cols,
        "standardization": "feature standard deviations from the held-out test set; zero standard deviations replaced by 1",
        "false_positive_row_id": fp_row_id,
        "false_positive_distance": fp_distance,
        "false_negative_row_id": fn_row_id,
        "false_negative_distance": fn_distance,
        "tie_breaking": "smallest row index",
        "deterministic_selection": True,
        "contains_real_user_data": False,
        "contains_real_identity_fields": False,
    }
    (RESULTS_DIR / "error_case_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    representative_rows = []
    for case_name, row_id in [("False positive: Normal -> Attack", fp_row_id), ("False negative: Attack -> Normal", fn_row_id)]:
        if row_id is None:
            continue
        row = analyzed.loc[row_id]
        representative_rows.append(
            {
                "case": case_name,
                "row_id": int(row_id),
                "attack_type": row["attack_type"],
                "otp_requests": int(row["otp_requests"]),
                "failure_rate": float(row["failure_rate"]),
                "repeat_ip_ratio": float(row["repeat_ip_ratio"]),
                "prefix_concentration": float(row["prefix_concentration"]),
                "success_rate": float(row["success_rate"]),
                "selection_distance": fp_distance if case_name.startswith("False positive") else fn_distance,
            }
        )
    pd.DataFrame(representative_rows).to_csv(RESULTS_DIR / "representative_error_cases.csv", index=False)


def out_of_fold_scores(model, X_train: pd.DataFrame, y_train: pd.Series, features: list[str]) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_scores = np.zeros(len(y_train), dtype=float)
    y_array = y_train.to_numpy()
    for train_pos, val_pos in splitter.split(X_train[features], y_array):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[train_pos][features], y_train.iloc[train_pos])
        fold_scores = model_scores(fold_model, X_train.iloc[val_pos][features])
        if fold_scores is None:
            raise ValueError("Cost-sensitive threshold selection requires model scores.")
        oof_scores[val_pos] = fold_scores
    return oof_scores


def threshold_metrics(y_true, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (y_score >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "false_positives": int(((y_true == 0) & (pred == 1)).sum()),
        "false_negatives": int(((y_true == 1) & (pred == 0)).sum()),
    }


def select_cost_sensitive_thresholds(y_train: pd.Series, oof_score: np.ndarray) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
    selection_rows = []
    for threshold in thresholds:
        metrics = threshold_metrics(y_train, oof_score, float(threshold))
        selection_rows.append({"threshold": float(threshold), **metrics})
    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(RESULTS_DIR / "threshold_selection_oof.csv", index=False)

    cost_rows = []
    for scenario, fn_cost, fp_cost in [
        ("FN:FP = 1:1", 1, 1),
        ("FN:FP = 5:1", 5, 1),
        ("FN:FP = 10:1", 10, 1),
        ("FN:FP = 1:5", 1, 5),
    ]:
        scored = selection_df.copy()
        scored["training_oof_relative_cost"] = (
            scored["false_negatives"] * fn_cost
            + scored["false_positives"] * fp_cost
        )
        scored["distance_to_0_5"] = (scored["threshold"] - 0.5).abs()
        best = scored.sort_values(
            ["training_oof_relative_cost", "distance_to_0_5", "threshold"],
            ascending=[True, True, False],
        ).iloc[0]
        cost_rows.append(
            {
                "cost_scenario": scenario,
                "false_negative_cost": fn_cost,
                "false_positive_cost": fp_cost,
                "selected_threshold": float(best["threshold"]),
                "training_oof_relative_cost": int(best["training_oof_relative_cost"]),
                "selection_method": "five_fold_training_oof_min_cost",
                "tie_breaking": "closest_to_0.5_then_higher_threshold",
            }
        )
    return pd.DataFrame(cost_rows)


def write_curve_and_threshold_outputs(
    y_test,
    y_score: np.ndarray,
    y_train: pd.Series,
    oof_score: np.ndarray,
) -> None:
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_score)

    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
        RESULTS_DIR / "roc_curve.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    ).to_csv(RESULTS_DIR / "pr_curve.csv", index=False)

    threshold_rows = []
    for threshold in np.round(np.arange(0.1, 1.0, 0.1), 1):
        metrics = threshold_metrics(y_test, y_score, float(threshold))
        threshold_rows.append(
            {
                "threshold": float(threshold),
                **metrics,
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(RESULTS_DIR / "threshold_tradeoff.csv", index=False)

    selected_thresholds = select_cost_sensitive_thresholds(y_train, oof_score)
    test_cost_rows = []
    for _, row in selected_thresholds.iterrows():
        metrics = threshold_metrics(y_test, y_score, float(row["selected_threshold"]))
        relative_cost = (
            metrics["false_negatives"] * int(row["false_negative_cost"])
            + metrics["false_positives"] * int(row["false_positive_cost"])
        )
        test_cost_rows.append(
            {
                **row.to_dict(),
                **metrics,
                "heldout_relative_cost": int(relative_cost),
            }
        )
    pd.DataFrame(test_cost_rows).to_csv(RESULTS_DIR / "cost_sensitive_thresholds.csv", index=False)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label="Random Forest")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=200)
    plt.close()


def write_prevalence_sensitivity(model_name: str, y_true: pd.Series, y_score: np.ndarray, benchmark_prevalence: float) -> pd.DataFrame:
    metrics = threshold_metrics(y_true, y_score, 0.5)
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    tpr = metrics["recall"]
    fpr = metrics["false_positives"] / negatives if negatives else 0.0
    rows = []
    for prevalence in [0.001, 0.01, 0.05, 0.10, benchmark_prevalence]:
        denominator = tpr * prevalence + fpr * (1 - prevalence)
        adjusted_precision = (tpr * prevalence / denominator) if denominator else 0.0
        rows.append(
            {
                "model": model_name,
                "assumed_attack_prevalence": prevalence,
                "tpr": tpr,
                "fpr": fpr,
                "adjusted_precision": adjusted_precision,
                "source": "mathematical projection from seed-42 held-out TPR and FPR",
                "deployment_estimate": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "prevalence_sensitivity.csv", index=False)
    return out


def evaluate_random_forest_and_rule(df: pd.DataFrame, feature_cols: list[str], seed: int) -> tuple[dict, dict]:
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )
    X_train = df.loc[train_idx, feature_cols]
    X_test = df.loc[test_idx, feature_cols]
    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    rf_result = evaluate_model("Random Forest", model, X_train, X_test, y_train, y_test, feature_cols)

    rule = fit_rule_baseline(X_train, y_train)
    rule_pred = apply_rule_baseline(X_test, rule)
    rule_result = prediction_metrics("Tuned Rule Baseline", "domain_rules", 4, y_test, rule_pred)
    return rf_result, rule_result


def run_difficulty_evaluation(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for difficulty in ["easy", "overlap", "adaptive"]:
        variant = make_difficulty_variant(df, difficulty, RANDOM_SEED)
        rf_result, rule_result = evaluate_random_forest_and_rule(variant, feature_cols, RANDOM_SEED)
        for result in [rf_result, rule_result]:
            result["difficulty"] = difficulty
            rows.append(result)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "difficulty_metrics.csv", index=False)

    rf_only = out[out["model"] == "Random Forest"]
    plt.figure(figsize=(5.5, 3.8))
    plt.plot(rf_only["difficulty"], rf_only["f1"], marker="o", label="Random Forest F1")
    plt.plot(rf_only["difficulty"], rf_only["recall"], marker="s", label="Random Forest recall")
    plt.xlabel("Benchmark difficulty")
    plt.ylabel("Score")
    plt.title("Difficulty Progression")
    plt.ylim(0.75, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "difficulty_progression.png", dpi=200)
    plt.close()
    return out


def run_attack_intensity_evaluation(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for intensity in [0.2, 0.4, 0.6, 0.8, 1.0]:
        variant = make_attack_intensity_variant(df, intensity, RANDOM_SEED)
        rf_result, rule_result = evaluate_random_forest_and_rule(variant, feature_cols, RANDOM_SEED)
        for result in [rf_result, rule_result]:
            result["attack_intensity"] = intensity
            rows.append(result)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "attack_intensity_metrics.csv", index=False)

    plt.figure(figsize=(5.8, 4.1))
    for model_name, group in out.groupby("model"):
        plt.plot(group["attack_intensity"], group["recall"], marker="o", linewidth=1.8, markersize=5.2, label=model_name)
    plt.xlabel("Attack intensity", fontsize=10)
    plt.ylabel("Recall", fontsize=10)
    plt.title("Attack Intensity Sensitivity", fontsize=11)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attack_intensity_recall.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "attack_intensity_recall.pdf", bbox_inches="tight")
    plt.close()
    return out


def run_cross_generator_evaluation(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx,
    test_idx,
) -> pd.DataFrame:
    heldout_base = df.loc[test_idx].copy()
    shifted = make_generator_shift_variant(heldout_base, RANDOM_SEED)
    X_train = df.loc[train_idx, feature_cols]
    y_train = df.loc[train_idx, "label"]
    X_test = shifted[feature_cols]
    y_test = shifted["label"]

    metadata = {
        "source_dataset": "otpfloodguard_simulated_windows.csv",
        "training_generator": "overlap_v1",
        "training_generator_seed": RANDOM_SEED,
        "training_split_seed": RANDOM_SEED,
        "shifted_generator": "shifted_v2",
        "shift_seed": RANDOM_SEED + 707,
        "shifted_generator_seed": RANDOM_SEED + 707,
        "shifted_test_size": int(len(shifted)),
        "class_counts": {str(k): int(v) for k, v in shifted["label"].value_counts().sort_index().items()},
        "attack_subtype_counts": {str(k): int(v) for k, v in shifted["attack_type"].value_counts().sort_index().items()},
        "shifted_data_construction": "held-out test rows transformed under shifted feature generator",
        "label_preserving_transformation": True,
        "transformed_feature_groups": [
            "benign burst ambiguity",
            "attack request intensity",
            "infrastructure reuse and distribution",
            "timing and multiplicative feature noise",
        ],
        "training_rows_used_in_shifted_test": False,
        "model_refit_on_shifted_data": False,
        "scaler_refit_on_shifted_data": False,
        "feature_reranking_on_shifted_data": False,
        "threshold_retuning_on_shifted_data": False,
        "rule_retuning_on_shifted_data": False,
    }
    (RESULTS_DIR / "generator_shift_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        ),
    }
    rows = []
    for model_name, model in models.items():
        result = evaluate_model(model_name, model, X_train, X_test, y_train, y_test, feature_cols)
        result["train_generator"] = "overlap_v1"
        result["test_generator"] = "shifted_v2"
        rows.append(result)

    rule = fit_rule_baseline(X_train, y_train)
    rule_pred = apply_rule_baseline(X_test, rule)
    rule_result = prediction_metrics("Tuned Rule Baseline", "domain_rules", 4, y_test, rule_pred)
    rule_result["train_generator"] = "overlap_v1"
    rule_result["test_generator"] = "shifted_v2"
    rows.append(rule_result)

    vf_rule = fit_velocity_failure_rule(X_train, y_train)
    vf_pred = apply_velocity_failure_rule(X_test, vf_rule)
    vf_result = prediction_metrics("Velocity+Failure Rule", "velocity_failure_rules", 2, y_test, vf_pred)
    vf_result["train_generator"] = "overlap_v1"
    vf_result["test_generator"] = "shifted_v2"
    rows.append(vf_result)

    out = pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=False)
    out.to_csv(RESULTS_DIR / "cross_generator_metrics.csv", index=False)
    return out


def run_permutation_importance_cv(model, X_train: pd.DataFrame, y_train: pd.Series, features: list[str]) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    y_array = y_train.to_numpy()
    for fold, (train_pos, val_pos) in enumerate(splitter.split(X_train[features], y_array), start=1):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[train_pos][features], y_train.iloc[train_pos])
        result = permutation_importance(
            fold_model,
            X_train.iloc[val_pos][features],
            y_train.iloc[val_pos],
            n_repeats=8,
            random_state=RANDOM_SEED + fold,
            scoring="f1",
            n_jobs=-1,
        )
        for feature, mean, std in zip(features, result.importances_mean, result.importances_std):
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "fold_importance_mean": mean,
                    "fold_importance_std": std,
                    "n_folds": 5,
                    "n_repeats": 8,
                    "scoring_metric": "f1",
                    "heldout_test_used": False,
                }
            )
    fold_df = pd.DataFrame(rows)
    out = (
        fold_df.groupby("feature")
        .agg(
            importance_mean=("fold_importance_mean", "mean"),
            importance_std=("fold_importance_mean", "std"),
            n_folds=("n_folds", "first"),
            n_repeats=("n_repeats", "first"),
            scoring_metric=("scoring_metric", "first"),
            heldout_test_used=("heldout_test_used", "first"),
        )
        .reset_index()
        .sort_values("importance_mean", ascending=False)
    )
    out.to_csv(RESULTS_DIR / "permutation_importance_cv.csv", index=False)
    out.to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
    return out


def write_sanity_checks(df: pd.DataFrame, feature_cols: list[str]) -> None:
    sample = df.sample(n=min(2500, len(df)), random_state=RANDOM_SEED)

    plt.figure(figsize=(5.8, 4.35))
    for label_value, label_name, color in [
        (0, "Normal windows", "#2B6CB0"),
        (1, "Attack windows", "#C2410C"),
    ]:
        group = sample[sample["label"] == label_value]
        plt.scatter(
            group["request_velocity_per_sec"],
            group["failure_rate"],
            c=color,
            label=label_name,
            alpha=0.30,
            s=13,
            edgecolors="none",
        )
    plt.xlabel("Request velocity per second", fontsize=10)
    plt.ylabel("Failure rate", fontsize=10)
    plt.title("Why the Benchmark Is Not Trivially Separable", fontsize=11)
    overlap = Ellipse(
        xy=(0.45, 0.55),
        width=0.55,
        height=0.48,
        angle=8,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="#111827",
        alpha=0.85,
    )
    plt.gca().add_patch(overlap)
    plt.annotate(
        "Overlap region:\nhard normal + low-intensity attack",
        xy=(0.45, 0.55),
        xytext=(0.86, 0.30),
        arrowprops=dict(arrowstyle="->", color="#111827", lw=1.4),
        fontsize=8.8,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", alpha=0.9),
    )
    plt.grid(True, alpha=0.2)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(frameon=True, fontsize=8.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_overlap_check.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "class_overlap_check.pdf", bbox_inches="tight")
    plt.close()

    corr_features = [
        "otp_requests",
        "failure_rate",
        "success_rate",
        "prefix_concentration",
        "repeat_ip_ratio",
        "risk_country_ratio",
        "avg_interarrival_ms",
    ]
    corr = df[corr_features].corr()
    corr.index.name = "feature"
    corr.to_csv(RESULTS_DIR / "feature_correlation.csv")
    plt.figure(figsize=(6.6, 5.35))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr_features)), corr_features, rotation=42, ha="right", fontsize=8)
    plt.yticks(range(len(corr_features)), corr_features, fontsize=8)
    plt.title("Selected Feature Correlation Sanity Check", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlation_check.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "feature_correlation_check.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10.4, 4.35))
    ax = plt.gca()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def rounded_box(x, y, w, h, title, body, fc="#F8FAFC", ec="#174A7C", lw=1.4, title_fs=9.0, body_fs=7.6):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h * 0.70,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
            color="#111827",
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.36,
            body,
            ha="center",
            va="center",
            fontsize=body_fs,
            linespacing=1.12,
            color="#111827",
            zorder=4,
        )

    def arrow(start, end, rad=0.0, color="#0F3F6B", lw=2.0, scale=16):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=5,
        )
        ax.add_patch(patch)

    rounded_box(
        0.035,
        0.58,
        0.19,
        0.25,
        "Public Evidence",
        "OTP/SMS abuse reports\nMITRE / Prelude\ntelecom pumping studies",
        fc="#F8FAFC",
    )
    rounded_box(
        0.035,
        0.23,
        0.19,
        0.25,
        "Task Scope",
        "server-side request windows\nno SMS content\nno private raw logs",
        fc="#F8FAFC",
    )

    core = FancyBboxPatch(
        (0.285, 0.17),
        0.39,
        0.70,
        boxstyle="round,pad=0.018,rounding_size=0.024",
        fc="#EEF6FF",
        ec="#0F3F6B",
        lw=1.9,
        zorder=1,
    )
    ax.add_patch(core)
    ax.text(
        0.48,
        0.81,
        "OTPFloodGuard Benchmark Core",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#0F172A",
        zorder=4,
    )
    rounded_box(0.315, 0.61, 0.145, 0.14, "Threat Model", "A1 flooder\nA2 pumping\nA3 adaptive", fc="#FFFFFF", title_fs=9.0, body_fs=7.15)
    rounded_box(0.500, 0.61, 0.145, 0.14, "Feature Map", "20 behavioral\nfeatures", fc="#FFFFFF", title_fs=9.0, body_fs=7.4)
    rounded_box(0.315, 0.36, 0.145, 0.14, "Generator", "normal bursts\nflooding / pumping\nsequential spray", fc="#FFFFFF", title_fs=9.0, body_fs=7.15)
    rounded_box(0.500, 0.36, 0.145, 0.14, "Controls", "difficulty regimes\nattack intensity\ngenerator shift", fc="#FFFFFF", title_fs=9.0, body_fs=7.15)
    rounded_box(0.405, 0.20, 0.150, 0.105, "Replaceability", "replace with\naggregate logs", fc="#FFFFFF", title_fs=8.9, body_fs=7.1)

    rounded_box(
        0.735,
        0.58,
        0.225,
        0.25,
        "Lightweight Baselines",
        "rules / Logistic Regression\nRandom Forest\nGradient Boosting",
        fc="#F8FAFC",
    )
    rounded_box(
        0.735,
        0.23,
        0.225,
        0.25,
        "Evaluation Outputs",
        "multi-split + ablation\ngenerator-shift robustness\nerrors + cost thresholds",
        fc="#F8FAFC",
    )

    arrow((0.225, 0.705), (0.285, 0.705), lw=2.3, scale=19)
    arrow((0.225, 0.355), (0.285, 0.355), lw=2.3, scale=19)
    arrow((0.675, 0.705), (0.735, 0.705), lw=2.3, scale=19)
    arrow((0.675, 0.355), (0.735, 0.355), lw=2.3, scale=19)
    arrow((0.460, 0.68), (0.500, 0.68), lw=1.7, scale=14)
    arrow((0.388, 0.61), (0.388, 0.50), lw=1.7, scale=14)
    arrow((0.573, 0.61), (0.573, 0.50), lw=1.7, scale=14)
    arrow((0.460, 0.43), (0.500, 0.43), lw=1.7, scale=14)
    arrow((0.480, 0.36), (0.480, 0.305), lw=1.7, scale=14)

    ax.text(
        0.50,
        0.06,
        "Architecture view: evidence-constrained assumptions enter a replaceable benchmark core, then feed stress-tested baseline evaluation.",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#374151",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "benchmark_pipeline.png", dpi=200)
    plt.close()

    soc_modules = [
        ("OTP Logs", "server-side request metadata"),
        ("Window Aggregation", "60-second behavioral features"),
        ("OTPFloodGuard Score", "lightweight model or rule score"),
        ("SOC Action", "alert / review / challenge\nnot automatic blocking"),
    ]
    plt.figure(figsize=(9.2, 2.8))
    ax = plt.gca()
    ax.axis("off")
    x_positions = np.linspace(0.12, 0.88, len(soc_modules))
    for idx, (x, (title, body)) in enumerate(zip(x_positions, soc_modules)):
        ax.text(
            x,
            0.55,
            f"{title}\n\n{body}",
            ha="center",
            va="center",
            fontsize=8.6,
            linespacing=1.18,
            bbox=dict(boxstyle="round,pad=0.45", fc="#ECFDF5", ec="#047857", lw=1.1),
        )
        if idx < len(soc_modules) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.10, 0.55),
                xytext=(x + 0.10, 0.55),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#047857"),
            )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "soc_usage_mode.png", dpi=200)
    plt.close()


def run_quick_mode() -> None:
    """Run a fast seed-42 quick verification without overwriting full-result tables."""
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = build_dataset()
    feature_cols = [c for c in df.columns if c not in {"attack_type", "label"}]
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    X_train = df.loc[train_idx, feature_cols]
    X_test = df.loc[test_idx, feature_cols]
    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }

    rows = []
    for model_name, model in models.items():
        rows.append(evaluate_model(model_name, model, X_train, X_test, y_train, y_test, feature_cols))

    rule = fit_rule_baseline(X_train, y_train)
    rule_pred = apply_rule_baseline(X_test, rule)
    rows.append(prediction_metrics("Tuned Rule Baseline", "domain_rules", 4, y_test, rule_pred))

    vf_rule = fit_velocity_failure_rule(X_train, y_train)
    vf_pred = apply_velocity_failure_rule(X_test, vf_rule)
    rows.append(prediction_metrics("Velocity+Failure Rule", "velocity_failure_rules", 2, y_test, vf_pred))

    quick_metrics = pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=False)
    quick_metrics.to_csv(RESULTS_DIR / "quick_metrics.csv", index=False)
    summary = {
        "mode": "quick",
        "note": "Fast seed-42 quick verification; full paper tables require running without --quick.",
        "samples": int(len(df)),
        "features": len(feature_cols),
        "best_model": str(quick_metrics.iloc[0]["model"]),
        "best_f1": float(quick_metrics.iloc[0]["f1"]),
        "output": str(RESULTS_DIR / "quick_metrics.csv"),
    }
    (RESULTS_DIR / "quick_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    for stale_name in STALE_RESULT_FILES:
        stale_path = RESULTS_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()
    write_environment_config()
    write_reporting_precision_config()

    df = build_dataset()
    data_path = DATA_DIR / "otpfloodguard_simulated_windows.csv"
    df.to_csv(data_path, index=False)

    feature_cols = [c for c in df.columns if c not in {"attack_type", "label"}]
    write_feature_dictionary(df, feature_cols)
    write_benchmark_config(df)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    X_train = df.loc[train_idx, feature_cols]
    X_test = df.loc[test_idx, feature_cols]
    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]

    selector = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)
    importances = pd.DataFrame(
        {"feature": feature_cols, "importance": selector.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    write_feature_ranking_metadata(importances["feature"].tolist())

    feature_sets = {
        "full": feature_cols,
        "top_5": importances.head(5)["feature"].tolist(),
        "top_10": importances.head(10)["feature"].tolist(),
        "top_15": importances.head(15)["feature"].tolist(),
    }

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }
    write_model_config(models, feature_count=len(feature_cols), probability_threshold=0.5)

    rows = []
    for feature_set_name, features in feature_sets.items():
        for model_name, model in models.items():
            result = evaluate_model(model_name, model, X_train, X_test, y_train, y_test, features)
            result["feature_set"] = feature_set_name
            rows.append(result)

    rule = fit_rule_baseline(X_train, y_train)
    rule_pred = apply_rule_baseline(X_test, rule)
    rows.append(prediction_metrics("Tuned Rule Baseline", "domain_rules", 4, y_test, rule_pred))
    (RESULTS_DIR / "rule_baseline.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")

    velocity_failure_rule = fit_velocity_failure_rule(X_train, y_train)
    velocity_failure_pred = apply_velocity_failure_rule(X_test, velocity_failure_rule)
    rows.append(prediction_metrics("Velocity+Failure Rule", "velocity_failure_rules", 2, y_test, velocity_failure_pred))
    (RESULTS_DIR / "velocity_failure_rule.json").write_text(json.dumps(velocity_failure_rule, indent=2), encoding="utf-8")
    write_rule_config(rule, velocity_failure_rule)

    metrics = pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=False)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    best = metrics.iloc[0]
    best_features = feature_sets[str(best["feature_set"])]
    best_model = models[str(best["model"])]
    best_model.fit(X_train[best_features], y_train)
    y_pred = best_model.predict(X_test[best_features])
    y_score = model_scores(best_model, X_test[best_features])
    if y_score is not None:
        oof_score = out_of_fold_scores(best_model, X_train, y_train, best_features)
        write_curve_and_threshold_outputs(y_test, y_score, y_train, oof_score)
        prevalence_sensitivity = write_prevalence_sensitivity(str(best["model"]), y_test, y_score, float(df["label"].mean()))
    else:
        prevalence_sensitivity = pd.DataFrame()
    permutation = run_permutation_importance_cv(best_model, X_train, y_train, best_features)
    run_error_analysis(df.loc[test_idx].copy(), y_test, y_pred, feature_cols)
    ablation = run_ablation_study(X_train, X_test, y_train, y_test, feature_cols)
    seed_metrics = run_multi_split_evaluation(df, feature_cols)
    generator_seed_metrics = run_generator_seed_sensitivity(
        {str(k): int(v) for k, v in df["attack_type"].value_counts().sort_index().items()},
        feature_cols,
    )
    difficulty_metrics = run_difficulty_evaluation(df, feature_cols)
    intensity_metrics = run_attack_intensity_evaluation(df, feature_cols)
    cross_generator_metrics = run_cross_generator_evaluation(df, feature_cols, train_idx, test_idx)
    write_sanity_checks(df, feature_cols)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "attack"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {best['model']} ({best['feature_set']})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_best_model.png", dpi=200)
    plt.close()

    top_plot = importances.head(10).sort_values("importance")
    plt.figure(figsize=(8, 4.8))
    plt.barh(top_plot["feature"], top_plot["importance"])
    plt.xlabel("Random Forest importance")
    plt.title("Top 10 OTPFloodGuard Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top10_feature_importance.png", dpi=200)
    plt.close()

    sensitivity_rows = []
    for window_seconds in [30, 60, 120]:
        window_df = make_window_variant(df, window_seconds, RANDOM_SEED)
        window_feature_cols = [c for c in window_df.columns if c not in {"attack_type", "label"}]
        window_X_train, window_X_test, window_y_train, window_y_test = train_test_split(
            window_df[window_feature_cols],
            window_df["label"],
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=window_df["label"],
        )
        window_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        result = evaluate_model(
            "Random Forest",
            window_model,
            window_X_train,
            window_X_test,
            window_y_train,
            window_y_test,
            window_feature_cols,
        )
        result["window_seconds"] = window_seconds
        sensitivity_rows.append(result)

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(RESULTS_DIR / "window_sensitivity.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(sensitivity["window_seconds"], sensitivity["f1"], marker="o", label="F1-score")
    plt.plot(sensitivity["window_seconds"], sensitivity["recall"], marker="s", label="Recall")
    plt.xlabel("Window length (seconds)")
    plt.ylabel("Score")
    plt.title("Window Sensitivity of Random Forest")
    plt.ylim(0.90, 1.00)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "window_sensitivity.png", dpi=200)
    plt.close()

    rf_seed_metrics = seed_metrics[seed_metrics["model"] == "Random Forest"]
    rf_generator_seed_metrics = generator_seed_metrics[generator_seed_metrics["model"] == "Random Forest"]
    summary = {
        "random_seed": RANDOM_SEED,
        "samples": int(len(df)),
        "features": len(feature_cols),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "attack_rate": float(df["label"].mean()),
        "normal_windows": int((df["label"] == 0).sum()),
        "attack_windows": int((df["label"] == 1).sum()),
        "best_model": str(best["model"]),
        "best_feature_set": str(best["feature_set"]),
        "best_f1": float(best["f1"]),
        "best_recall": float(best["recall"]),
        "best_roc_auc": float(best["roc_auc"]),
        "best_pr_auc": float(best["pr_auc"]),
        "rule_baseline_f1": float(metrics.loc[metrics["model"] == "Tuned Rule Baseline", "f1"].iloc[0]),
        "velocity_failure_rule_f1": float(metrics.loc[metrics["model"] == "Velocity+Failure Rule", "f1"].iloc[0]),
        "multi_split_f1_mean": float(rf_seed_metrics["f1"].mean()),
        "multi_split_f1_std": float(rf_seed_metrics["f1"].std()),
        "generator_seed_f1_mean": float(rf_generator_seed_metrics["f1"].mean()),
        "generator_seed_f1_std": float(rf_generator_seed_metrics["f1"].std()),
        "worst_ablation": str(ablation.iloc[-1]["feature_set"]),
        "worst_ablation_f1": float(ablation.iloc[-1]["f1"]),
        "adaptive_rf_f1": float(difficulty_metrics[(difficulty_metrics["model"] == "Random Forest") & (difficulty_metrics["difficulty"] == "adaptive")]["f1"].iloc[0]),
        "low_intensity_rf_recall": float(intensity_metrics[(intensity_metrics["model"] == "Random Forest") & (intensity_metrics["attack_intensity"] == 0.2)]["recall"].iloc[0]),
        "cross_generator_rf_f1": float(cross_generator_metrics[cross_generator_metrics["model"] == "Random Forest"]["f1"].iloc[0]),
        "cross_generator_rf_recall": float(cross_generator_metrics[cross_generator_metrics["model"] == "Random Forest"]["recall"].iloc[0]),
        "top_permutation_feature": str(permutation.iloc[0]["feature"]),
        "prevalence_sensitivity_rows": int(len(prevalence_sensitivity)),
        "data_path": str(data_path),
    }
    (RESULTS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OTPFloodGuard benchmark results.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast seed-42 quick verification and write quick_metrics.csv without overwriting full tables.",
    )
    args = parser.parse_args()
    if args.quick:
        run_quick_mode()
    else:
        main()
