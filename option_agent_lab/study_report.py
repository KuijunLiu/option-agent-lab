"""Reports distinguish numerical evidence, pending agents, and paired results."""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .study import collect_agent_results, read_json, save_rows, verify_lock
from .search import write_json

COLORS = {"random": "#64748b", "sobol": "#2563eb",
          "financial_stress": "#7c3aed", "de": "#e67e22"}


def make_study_report(study):
    study = Path(study).resolve()
    verify_lock(study)
    config = read_json(study / "protocol.json")["config"]
    selection = read_json(study / "development_selection.json")
    seeds = config["model_seeds"]
    numerical, audit_rows, examples = [], [], []
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4.3),
                             squeeze=False, layout="constrained")
    delta_fig, delta_axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    rng = np.random.default_rng(10)
    subset_labels, subset_fractions = [], []
    model_colors = plt.get_cmap("tab10")
    for j, seed in enumerate(seeds):
        run = study / "models" / f"model_{seed}"
        summary = read_json(run / "benchmarks_v2/summary.json")
        for method in COLORS:
            selected_rows = [r for r in summary["runs"] if r["method"] == method]
            curves = []
            for row in selected_rows:
                result = dict(np.load(run / "benchmarks_v2" / f"{method}_{row['seed']}.npz"))
                curves.append(np.maximum.accumulate(result["error"]) * 100)
                numerical.append({"model_seed": seed, "search_seed": row["seed"], "method": method,
                    "max_price_error": row["max_error_normalized"] * 100,
                    "failing_bins": row["failure_regions"], "price_bound_violations": row["bound_violations"],
                    "numeric_seconds": row["seconds"], "evaluations": row["n"]})
            curves = np.asarray(curves)
            xs = np.arange(1, curves.shape[1] + 1)
            axes[0, j].plot(xs, curves.mean(axis=0), color=COLORS[method], label=method)
            axes[0, j].fill_between(xs, curves.min(axis=0), curves.max(axis=0), color=COLORS[method], alpha=.12)
        axes[0, j].set(title=f"Frozen model seed {seed}", xlabel="Evaluated points",
                       ylabel="Largest price error found (K=100)")
        axes[0, j].legend(fontsize=8)
        axes[0, j].grid(alpha=.2)
        audit = read_json(run / "greeks/audit.json")
        for split in ("test", "stress"):
            stats = audit["splits"][split]
            arrays = dict(np.load(run / "greeks" / f"{split}.npz"))
            good_price = arrays["price_error"] <= config["price_failure_threshold"]
            delta_bad = arrays["delta_error"] > config["delta_diagnostic_threshold"]
            fraction = float(delta_bad[good_price].mean()) if good_price.any() else None
            audit_rows.append({"model_seed": seed, "split": split, "n": stats["n"],
                "price_mae": stats["price"]["mae"] * 100,
                "price_max_error": stats["price"]["max_abs_error"] * 100,
                "delta_mae": stats["delta"]["mae"], "delta_max_error": stats["delta"]["max_abs_error"],
                "price_accurate_n": int(good_price.sum()),
                "delta_bad_given_price_accurate_fraction": fraction,
                "delta_bound_violations": stats["delta_outside_unit_interval_count"]})
            if good_price.any():
                indices = np.flatnonzero(good_price)
                i = indices[np.argmax(arrays["delta_error"][indices])]
                examples.append({"model_seed": seed, "split": split,
                    "m_T_sigma": arrays["X"][i].tolist(),
                    "price_prediction": float(arrays["pred_price"][i] * 100),
                    "price_reference": float(arrays["ref_price"][i] * 100),
                    "price_error": float(arrays["price_error"][i] * 100),
                    "delta_prediction": float(arrays["pred_delta"][i]),
                    "delta_reference": float(arrays["ref_delta"][i]),
                    "delta_error": float(arrays["delta_error"][i])})
            ix = rng.choice(len(good_price), min(1000, len(good_price)), replace=False)
            delta_axes[0].scatter(np.maximum(arrays["price_error"][ix] * 100, 1e-8),
                np.maximum(arrays["delta_error"][ix], 1e-8), s=5, alpha=.2,
                color=model_colors(j), label=f"model {seed}" if split == "test" else None)
            subset_labels.append(f"{seed} / {split}")
            subset_fractions.append(100 * fraction if fraction is not None else np.nan)
    fig.suptitle("Numerical searches: mean and min/max over three search seeds")
    fig.savefig(study / "search_comparison.png", dpi=160)
    plt.close(fig)
    delta_axes[0].set(xscale="log", yscale="log", xlabel="Absolute price error (K=100)",
                     ylabel="Absolute delta error", title="Independent test/stress samples")
    delta_axes[0].axvline(.1, color="black", linestyle="--", linewidth=1)
    delta_axes[0].axhline(.05, color="black", linestyle=":", linewidth=1)
    delta_axes[0].legend(fontsize=8)
    delta_axes[1].barh(subset_labels, subset_fractions, color="#7c3aed")
    delta_axes[1].set(xlabel="Delta error > 0.05 among price-accurate points (%)",
                     title="Price-accurate means price error <= 0.10")
    for ax in delta_axes:
        ax.grid(alpha=.15)
    delta_fig.savefig(study / "delta_diagnostics.png", dpi=160)
    plt.close(delta_fig)

    aggregates = []
    for method in COLORS:
        rows = [r for r in numerical if r["method"] == method]
        values = [r["max_price_error"] for r in rows]
        aggregates.append({"method": method, "runs": len(rows),
            "mean_max_price_error": float(np.mean(values)),
            "sd_max_price_error": float(np.std(values, ddof=1)) if len(values) > 1 else 0,
            "mean_failing_bins": float(np.mean([r["failing_bins"] for r in rows])),
            "mean_numeric_seconds": float(np.mean([r["numeric_seconds"] for r in rows]))})
    agents = collect_agent_results(study)
    pairs = []
    for m in seeds:
        for seed in config["search_seeds"]:
            both = {r["condition"]: r for r in agents if r["model_seed"] == m and r["search_seed"] == seed}
            if all(both[c]["status"] == "completed" for c in config["conditions"]):
                a, b = both["feedback"], both["no_feedback"]
                pairs.append({"model_seed": m, "search_seed": seed,
                    "feedback_max_price_error": a["max_error_normalized"] * 100,
                    "no_feedback_max_price_error": b["max_error_normalized"] * 100,
                    "paired_difference": (a["max_error_normalized"] - b["max_error_normalized"]) * 100})
    save_rows(study / "numerical_results.csv", numerical)
    save_rows(study / "greeks_results.csv", audit_rows)
    if pairs:
        save_rows(study / "paired_agent_results.csv", pairs)
    write_json(study / "summary.json", {"numerical": aggregates, "audit": audit_rows,
        "agent_status_counts": dict(Counter(r["status"] for r in agents)),
        "paired_agent_results": pairs, "agent_rows": agents, "diagnostic_examples": examples})
    md = ["# Erdos v2 实际实验报告", "",
        "数值结果由本次保存的逐点评估生成。价格单位均按 K=100 换算。",
        f"三个新网络、每个三个搜索种子；每种数值方法共 {len(seeds)*len(config['search_seeds'])} 次搜索，每次1024点。",
        f"差分进化种群仅在旧开发模型上从16/32/64中选择，选定 {selection['selected_population']} 后冻结。", "",
        "## 已完成的数值对照", "",
        "| 方法 | 次数 | 最大已发现价格误差：均值 ± SD | 失败网格数：均值 | 数值搜索秒数：均值 |",
        "|---|---:|---:|---:|---:|"]
    for r in aggregates:
        md.append(f"| {r['method']} | {r['runs']} | {r['mean_max_price_error']:.4f} ± {r['sd_max_price_error']:.4f} | {r['mean_failing_bins']:.1f} | {r['mean_numeric_seconds']:.4f} |")
    md += ["", "更大的已发现误差表示更会找问题，不是网络被改进。SD仅描述这九次运行的离散程度，不是置信区间；三个模型才是模型层面的重复。",
           "每个网络的逐次结果见 numerical_results.csv。固定金融压力方案使用先验知识，未根据新网络结果调整。",
           "", "![数值搜索比较](search_comparison.png)", "", "## 价格与 delta 的独立诊断", "",
           "| 模型种子 | 集合 | 价格 MAE | Delta MAE | 价格误差≤0.10 的点中 Delta 误差>0.05 的比例 |",
           "|---|---|---:|---:|---:|"]
    for r in audit_rows:
        fraction = r["delta_bad_given_price_accurate_fraction"]
        shown = f"{fraction:.2%}" if fraction is not None else "无符合条件样本"
        md.append(f"| {r['model_seed']} | {r['split']} | {r['price_mae']:.4f} | {r['delta_mae']:.4f} | {shown} |")
    md += ["", "0.10价格误差和0.05 Delta误差均为预先设置的研究阈值，不是行业标准。诊断集没有提供给agent；它不是价格/Delta误差联合优化实验，也不是对冲损益回测。",
           "", "![Delta诊断](delta_diagnostics.png)", "", "## LLM 对照状态", ""]
    counts = Counter(r["status"] for r in agents)
    md.append(f"计划 {len(agents)} 个会话（最多 {len(agents)*7} 次决策调用）。当前状态：{dict(counts)}；完整配对 {len(pairs)}/{len(agents)//2}。")
    if not pairs:
        md.append("尚无可比较的真实 Codex 配对结果，不能宣称反馈有效或无效。旧助手 pilot 不并入本实验。")
    else:
        md += ["", "| 网络 | 搜索种子 | 反馈组最大误差 | 无新反馈组最大误差 | 配对差值 |", "|---|---:|---:|---:|---:|"]
        for r in pairs:
            md.append(f"| {r['model_seed']} | {r['search_seed']} | {r['feedback_max_price_error']:.4f} | {r['no_feedback_max_price_error']:.4f} | {r['paired_difference']:.4f} |")
        if len(pairs) < len(agents)//2:
            md.append("配对尚不完整；上述仅为已完成配对，不能据此作全实验优劣结论。失败会话独立报告，不静默删除。")
    md += ["", "## 解释边界", "",
        "- 合成数据衡量对Black–Scholes的近似误差；不验证真实市场定价、盈利或市场适应能力。",
        "- 搜索目标是价格最大误差；Delta只作为独立诊断，未利用其结果改动选择方法。",
        "- 数值搜索时间不含加载/文件写入；LLM墙钟时间还包括等待与控制器开销，两者应分别报告。",
        "- 单次最坏已发现误差不是真实全域最坏误差；有限网格覆盖也不等于失败机制种类。",
        "- 公布的网络与诊断结果已可见。后续修改提示词、方法或训练配置，应建立新研究目录，不能继续称其为未见测试。",
        "- 当前Codex桥接通过过滤传入上下文隔离反馈；read-only不提供整个文件系统的强保密隔离。",
        "", "完整协议、学习路线和论文依据见 ../../docs/EXPERIMENT_PLAN.md 与 ../../docs/LEARNING_PATH.md。"]
    (study / "REPORT.md").write_text("\n".join(md) + "\n")
    return {"report": str(study / "REPORT.md"), "numerical": aggregates,
            "agent_status_counts": dict(counts), "completed_pairs": len(pairs)}
