#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

# =========================
# ユーティリティ
# =========================
def is_empty_cut(x):
    if pd.isna(x):
        return True
    return str(x).strip() == ""


def parse_flag(flag):
    if pd.isna(flag) or str(flag).strip() == "":
        return []
    return [t.strip() for t in str(flag).split(",") if t.strip()]


def add_iki(flag):
    tokens = parse_flag(flag)
    if "iki" not in tokens:
        tokens.append("iki")
    return ",".join(tokens)


def eprint(msg):
    # 仕様どおり「標準出力」に出す（stderrではない）
    print(msg)


def unique_output_path(path):
    path = Path(path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def default_output_path(prefix):
    timestamp = datetime.now().strftime("%m%d%H%M")
    return unique_output_path(f"{prefix}_{timestamp}.csv")


def find_duplicate_paths(paths):
    seen = {}
    duplicates = []
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen:
            duplicates.append((seen[resolved], path))
        else:
            seen[resolved] = path
    return duplicates


# =========================
# メイン処理
# =========================
def main():
    parser = argparse.ArgumentParser(description="samplecut 抽出処理")
    parser.add_argument("--data", required=True, help="data CSV のパス")
    parser.add_argument("--wari", required=True, nargs="+",
                        help="WARI_*.csv のパス（複数指定可）")
    parser.add_argument("--out", default=None, help="出力CSV（未指定なら out_月日時分.csv）")
    parser.add_argument("--shortage", default=None, help="不足一覧CSV（未指定なら shortage_月日時分.csv）")
    parser.add_argument("--seed", type=int, default=None,
                        help="乱数seed（未指定なら毎回ランダム）")
    args = parser.parse_args()
    duplicate_wari = find_duplicate_paths(args.wari)
    if duplicate_wari:
        print("注意: 同じ WARI ファイルが複数回指定されています。処理を中止します。")
        for first, duplicate in duplicate_wari:
            print(f"  {first} / {duplicate}")
        sys.exit(1)

    out_path = unique_output_path(args.out) if args.out else default_output_path("out")
    shortage_path = unique_output_path(args.shortage) if args.shortage else default_output_path("shortage")

    # =========================
    # データ読み込み
    # =========================
    df = pd.read_csv(args.data)

    required_cols = ["CUT", "SAMPLENUMBER", "CAT", "PANEL", "AREA", "AGE", "CHK"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"必須列不足: {c}")

    if "Flag" not in df.columns:
        df["Flag"] = ""

    eligible_mask = df["CUT"].apply(is_empty_cut)

    # =========================
    # WARI 読み込み
    # =========================
    wari_frames = []
    for path in args.wari:
        w = pd.read_csv(path)
        for c in ["CAT", "PANEL", "AREA", "AGE", "件数"]:
            if c not in w.columns:
                raise ValueError(f"{path} に必須列 {c} がありません")
        w["__src__"] = path
        wari_frames.append(w)

    wari_all = pd.concat(wari_frames, ignore_index=True)

    wari_map = {}
    for _, r in wari_all.iterrows():
        key = (int(r["CAT"]), int(r["PANEL"]), int(r["AREA"]), int(r["AGE"]))
        if key in wari_map:
            eprint(f"エラー: WARI重複（先勝ち） key={key}")
            continue
        try:
            n = int(r["件数"])
            if n < 0:
                raise ValueError
        except Exception:
            eprint(f"エラー: 件数不正 src={r['__src__']} key={key} 件数={r['件数']} → 0扱い")
            n = 0
        wari_map[key] = n

    # =========================
    # WARI 未定義チェック
    # =========================
    eligible_keys = set(
        map(tuple, df.loc[eligible_mask, ["CAT", "PANEL", "AREA", "AGE"]].values)
    )
    for k in eligible_keys:
        if k not in wari_map:
            eprint(f"エラー: WARI未定義 key={k}（要求件数=0扱い）")
            wari_map[k] = 0

    # =========================
    # 抽出処理
    # =========================
    rng = np.random.default_rng(args.seed)
    shortages = []

    for (cat, panel, area, age), req in wari_map.items():
        if req <= 0:
            continue

        cond = (
            eligible_mask
            & (df["CAT"] == cat)
            & (df["PANEL"] == panel)
            & (df["AREA"] == area)
            & (df["AGE"] == age)
        )
        candidates = df[cond]

        picked = []
        remain = req

        for chk in [0, 1, 2]:
            idx = candidates[candidates["CHK"] == chk].index.difference(picked)
            if idx.empty:
                continue
            take = min(len(idx), remain)
            chosen = rng.choice(idx, size=take, replace=False)
            picked.extend(chosen)
            remain -= take
            if remain == 0:
                break

        if remain > 0:
            idx = candidates[candidates["CHK"] >= 3].index.difference(picked)
            if not idx.empty:
                take = min(len(idx), remain)
                chosen = rng.choice(idx, size=take, replace=False)
                picked.extend(chosen)
                remain -= take

        for i in picked:
            df.at[i, "Flag"] = add_iki(df.at[i, "Flag"])

        if remain > 0:
            shortages.append([cat, panel, area, age, req, req - remain, remain])

    # =========================
    # 出力
    # =========================
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    short_df = pd.DataFrame(
        shortages,
        columns=["CAT", "PANEL", "AREA", "AGE", "要求件数", "抽出件数", "不足件数"],
    )
    short_df.to_csv(shortage_path, index=False, encoding="utf-8-sig")

    print(f"out: {out_path}")
    print(f"shortage: {shortage_path}")


if __name__ == "__main__":
    main()
