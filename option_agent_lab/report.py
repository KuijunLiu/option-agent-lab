"""Generate figures and a factual report from saved observations."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import THRESHOLD
from .evaluation import concatenate, summarize


def make_report(run):
    run = Path(run)
    training = json.loads((run/"training_metrics.json").read_text())
    audit = json.loads((run/"audit.json").read_text())
    baselines = json.loads((run/"baselines/summary.json").read_text())
    groups = {}
    for row in baselines["runs"]:
        groups.setdefault(row["method"],[]).append(row)
    fig, axes = plt.subplots(1,2,figsize=(12,4.3),layout="constrained")
    colors = {"random":"#64748b","sobol":"#2563eb","differential_evolution":"#e67e22"}
    table = []
    for method,rows in groups.items():
        curves=[]
        for row in rows:
            result=dict(np.load(run/f"baselines/{method}_{row['seed']}.npz"))
            curves.append(np.maximum.accumulate(result["error"])*100)
        mean=np.mean(curves,axis=0)
        axes[0].plot(np.arange(1,len(mean)+1),mean,label=method,color=colors[method])
        axes[0].fill_between(np.arange(1,len(mean)+1),np.min(curves,axis=0),np.max(curves,axis=0),alpha=.12,color=colors[method])
        maximum=np.array([r["max_error_normalized"]*100 for r in rows])
        table.append({"method":method,"repeats":len(rows),"max_error_price_mean":float(maximum.mean()),
            "max_error_price_std":float(maximum.std(ddof=1)) if len(rows)>1 else 0,
            "failure_regions_mean":float(np.mean([r["failure_regions"] for r in rows])),
            "seconds_mean":float(np.mean([r["seconds"] for r in rows]))})
    pilot_lines=[]
    for folder in sorted((run/"sessions").glob("*")) if (run/"sessions").exists() else []:
        state=json.loads((folder/"state.json").read_text())
        if not state["round"]: continue
        result=concatenate([dict(np.load(folder/f"round_{i:03d}.npz")) for i in range(1,state["round"]+1)])
        axes[0].plot(np.arange(1,len(result["error"])+1),np.maximum.accumulate(result["error"])*100,
            label=f"{folder.name} (one pilot)",linestyle="--",linewidth=2)
        measured=summarize(result)
        pilot_lines.append(f"- {folder.name}：{state['used']} 个测试点，最大价格误差 {result['error'].max()*100:.6f}，发现 {measured['failure_regions']} 个失败区域，{measured['bound_violations']} 个点违反价格界限。仅一次 pilot，未重复验证统计优势。")
    with (run/"training_history.csv").open() as f: history=list(csv.DictReader(f))
    epochs=[int(r["epoch"]) for r in history]
    for key in ("train_mse","validation_mse"):
        if key in history[0]: axes[1].semilogy(epochs,[float(r[key]) for r in history],label=key)
    axes[0].set(xlabel="Cumulative evaluated parameter points",ylabel="Largest absolute price error found (K=100)",title="Search performance; bands = min/max over seeds")
    axes[1].set(xlabel="Training epoch",ylabel="MSE in normalized C/K units",title="Training and validation only")
    for ax in axes: ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(run/"overview.png",dpi=160)
    plt.close(fig)
    (run/"comparison.json").write_text(json.dumps(table,indent=2)+"\n")
    md=["# 实际运行报告","","本报告由保存的数值结果自动生成。价格误差按 K=100 换算。",
        f"本次 CPU 训练 {training['epochs_run']} 个 epoch，训练计时 {training['train_seconds']:.2f} 秒（不含环境安装与数据生成）。", "",
        "## 独立审计集", "", "| 数据集 | 样本数 | 平均绝对价格误差 | 最大价格误差 | 超阈值比例 |", "|---|---:|---:|---:|---:|"]
    for name,row in audit.items():
        md.append(f"| {name} | {row['n']} | {row['mae_normalized']*100:.6f} | {row['max_error_normalized']*100:.6f} | {row['failure_fraction']:.2%} |")
    md += ["", "## 搜索基线", "", "所有方法使用相同冻结网络、相同参数域、相同前128个Sobol点和相同总评估预算。", "",
        "| 方法 | 重复次数 | 最大价格误差 mean ± sample SD | 发现失败区域数 mean | 搜索时间 mean (s) |", "|---|---:|---:|---:|---:|"]
    for row in table:
        md.append(f"| {row['method']} | {row['repeats']} | {row['max_error_price_mean']:.6f} ± {row['max_error_price_std']:.6f} | {row['failure_regions_mean']:.1f} | {row['seconds_mean']:.4f} |")
    md += ["", "最大误差越大表示搜索更会发现错误；不是网络变差或变好。优化最大误差与覆盖更多不同区域是不同目标。",
        "失败区域按预先固定的6×5×4网格去重，共120格。图中基线实线是5次搜索的均值、阴影为最小/最大值；助手虚线为单次结果。", "",
        "## Agent 状态", "", *(pilot_lines or ["尚无真实反馈式agent会话；不能声称agent实验完成。"]), "",
        "## 限制", "", f"- 失败阈值预先设为 |prediction-reference|/K > {THRESHOLD}，即本例0.10价格单位；不是行业统一风险阈值。",
        "- 只有一个训练种子的网络；基线重复只反映搜索随机性，不代表跨模型泛化。",
        "- stress集是独立设计的短期限/平值诊断分布，不是随机市场发生概率，也未用于搜索反馈。",
        "- 神经网络测试的是对Black–Scholes的逼近；不能验证Black–Scholes的真实市场适用性。",
        "- 会话中的region+Sobol是混合方法。助手引导pilot不是重复的Codex CLI统计实验。",
        "- Codex CLI需要用户本机登录后验证；当前交付环境没有执行该登录路径。",
        "- baseline时间含数值搜索，不含训练/首次加载；session时间含加载，不能与baseline当作端到端成本直接比较。",
        "", "![运行图](overview.png)", "", "训练配置和耗时见 training_metrics.json；完整逐点评估见 baselines/ 和 sessions/。"]
    (run/"REPORT.md").write_text("\n".join(md)+"\n")
    return {"report":str(run/"REPORT.md"),"figure":str(run/"overview.png"),"comparison":table}
