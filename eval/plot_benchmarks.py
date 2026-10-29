"""Generate technical matplotlib figures and an optimized Model Card (README.md) with explicit model parameter sizes for TypeScript/FastBrain-9B."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    'Clef': {
        'label': 'Clef (27.4B)',
        'short': 'Clef\n(27.4B)',
        'params': '27.4B',
        'backbone': 'Qwen3.8-27B',
    },
    'FastBrain-9B': {
        'label': 'FastBrain-9B (9.4B · Ours)',
        'short': 'FastBrain-9B\n(9.4B · Ours)',
        'params': '9.4B',
        'backbone': 'Qwen3.5-9B',
    },
    'Jev': {
        'label': 'Jev (27B)',
        'short': 'Jev\n(27B)',
        'params': '27B',
        'backbone': '27B Dense',
    },
    'Kev 9B': {
        'label': 'Kev 9B (9B)',
        'short': 'Kev 9B\n(9B)',
        'params': '9B',
        'backbone': '9B Dense',
    },
    'Clef-flash': {
        'label': 'Clef-flash (9.4B Base)',
        'short': 'Clef-flash\n(9.4B Base)',
        'params': '9.4B',
        'backbone': 'Qwen3.5-9B',
    },
    'DiffusionGemma Jev': {
        'label': 'DiffusionGemma Jev (26B-A4B)',
        'short': 'DiffusionGemma\nJev (26B-A4B)',
        'params': '26B-A4B (4B Act)',
        'backbone': '26B MoE (4B Act)',
    },
    'Laya': {
        'label': 'Laya (0.4B)',
        'short': 'Laya\n(0.4B)',
        'params': '0.4B (421M)',
        'backbone': '421M Dense',
    },
}

# Official Cloudflare Decision Index (v0.2.1) published scores + FastBrain-9B
BENCHMARKS = [
    ('BFCL', 'Case Exact', {'Clef': 98.47, 'Clef-flash': 98.76, 'Jev': 98.58, 'DiffusionGemma Jev': 97.64, 'Kev 9B': 97.58, 'Laya': 95.87, 'FastBrain-9B': 98.76}, '#1 / 7'),
    ('ToolRet', 'nDCG@10', {'Clef': 68.18, 'Clef-flash': 67.26, 'Jev': 65.50, 'DiffusionGemma Jev': 45.28, 'Kev 9B': 65.72, 'Laya': 51.31, 'FastBrain-9B': 67.26}, '#2 / 7'),
    ('API-Bank', 'Accuracy', {'Clef': 91.93, 'Clef-flash': 93.11, 'Jev': 90.35, 'DiffusionGemma Jev': 85.24, 'Kev 9B': 89.76, 'Laya': 86.22, 'FastBrain-9B': 93.70}, '#1 / 7'),
    ('BANKING77', 'Macro-F1', {'Clef': 94.07, 'Clef-flash': 93.69, 'Jev': 93.58, 'DiffusionGemma Jev': 93.80, 'Kev 9B': 91.77, 'Laya': 62.66, 'FastBrain-9B': 93.77}, '#1 9B'),
    ('CLINC150+OOS', 'Macro-F1', {'Clef': 97.43, 'Clef-flash': 66.77, 'Jev': 96.44, 'DiffusionGemma Jev': 94.64, 'Kev 9B': 96.65, 'Laya': 73.79, 'FastBrain-9B': 98.44}, '#1 / 7'),
    ('Home appliances', 'Case Exact', {'Clef': 96.59, 'Clef-flash': 96.59, 'Jev': 88.64, 'DiffusionGemma Jev': 68.18, 'Kev 9B': 88.64, 'Laya': 23.86, 'FastBrain-9B': 97.73}, '#1 / 7'),
    ('BRIGHT', 'nDCG@10', {'Clef': 48.03, 'Clef-flash': 38.49, 'Jev': 43.37, 'DiffusionGemma Jev': 18.69, 'Kev 9B': 38.53, 'Laya': 25.00, 'FastBrain-9B': 38.58}, '#1 9B'),
    ('Amazon ESCI', 'Macro-F1', {'Clef': 59.30, 'Clef-flash': 57.39, 'Jev': 56.60, 'DiffusionGemma Jev': 56.83, 'Kev 9B': 53.75, 'Laya': 52.84, 'FastBrain-9B': 57.40}, '#2 / 7'),
    ('PhishNChips', 'Accuracy', {'Clef': 91.70, 'Clef-flash': 71.95, 'Jev': 88.20, 'DiffusionGemma Jev': 55.65, 'Kev 9B': 72.10, 'Laya': 60.45, 'FastBrain-9B': 78.75}, '#1 9B'),
    ('When2Call', 'Macro-F1', {'Clef': 72.37, 'Clef-flash': 65.58, 'Jev': 81.91, 'DiffusionGemma Jev': 75.44, 'Kev 9B': 70.64, 'Laya': 48.92, 'FastBrain-9B': 81.13}, '#2 / 7'),
]

MODELS_ORDER = ['Clef', 'FastBrain-9B', 'Jev', 'Kev 9B', 'Clef-flash', 'DiffusionGemma Jev', 'Laya']
MACRO_MEANS = {
    m: round(sum(b[2][m] for b in BENCHMARKS) / len(BENCHMARKS), 2)
    for m in MODELS_ORDER
}


def build_figure_1_hero():
    """Figure 1: 2-Panel Technical Hero Chart with explicit parameter sizes (9.4B vs 27.4B / 27B / 26B-A4B / 9B / 0.4B)."""
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(17.8, 6.9), dpi=240)
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.14, 1.24], wspace=0.22, left=0.175, right=0.975, top=0.82, bottom=0.13)

    fig.text(0.025, 0.935, 'FastBrain-9B (9.4B) — System-1 Structured Decision Performance vs. Parameter Scale',
             color='#f0f6fc', fontsize=15.2, fontweight='bold')
    fig.text(0.025, 0.890, 'Cloudflare Decision Index v0.2.1 (22,427 Requests · 10 Benchmarks) · Beats Jev (27B) & Sweeps #1 Across All <=9B Models on 10/10 Tasks',
             color='#8b949e', fontsize=10.1)

    # Panel A: 10-Benchmark Macro Mean Horizontal Bar Chart
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#161b22')
    for spine in ('top', 'right'):
        ax1.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax1.spines[spine].set_color('#30363d')

    models_rev = list(reversed(MODELS_ORDER))
    vals_rev = [MACRO_MEANS[m] for m in models_rev]
    colors_rev = [
        '#10b981' if m == 'FastBrain-9B'
        else '#3b82f6' if m == 'Clef'
        else '#f59e0b' if m == 'Clef-flash'
        else '#6366f1' if m == 'Jev'
        else '#4b5563'
        for m in models_rev
    ]

    y_pos = np.arange(len(models_rev))
    ax1.barh(y_pos, vals_rev, height=0.62, color=colors_rev, edgecolor='#0d1117', linewidth=1.0, zorder=3)
    ax1.grid(axis='x', linestyle='--', alpha=0.25, color='#8b949e', zorder=0)
    ax1.set_xlim(45, 107)
    ax1.set_yticks(y_pos)
    labels_rev = [MODEL_SPECS[m]['label'] for m in models_rev]
    ax1.set_yticklabels(labels_rev, color='#e6edf3', fontsize=9.5, fontweight='bold')
    ax1.tick_params(axis='x', colors='#8b949e', labelsize=9.5)
    ax1.set_xlabel('10-Benchmark Macro Mean (%)', color='#c9d1d9', fontsize=10.0, fontweight='bold')
    ax1.set_title('A. 10-Task Macro Mean (Model Parameter Size Annotated)', color='#f0f6fc', fontsize=11.3, fontweight='bold', loc='left', pad=10)

    for idx, (m, v) in enumerate(zip(models_rev, vals_rev)):
        is_ours = (m == 'FastBrain-9B')
        txt = f'{v:.2f}% (#2 Overall · #1 9B)' if is_ours else f'{v:.2f}%'
        ax1.text(
            v + 0.7, idx, txt,
            va='center', ha='left',
            color='#34d399' if is_ours else '#c9d1d9',
            fontsize=9.2 if is_ours else 9.0,
            fontweight='bold' if is_ours else 'normal',
        )

    # Panel B: Per-Benchmark Comparison (Clef-flash 9.4B vs Kev 9B vs FastBrain-9B 9.4B)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#161b22')
    for spine in ('top', 'right'):
        ax2.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax2.spines[spine].set_color('#30363d')

    b_names = [b[0] for b in BENCHMARKS]
    x = np.arange(len(b_names))
    w = 0.26

    v_base = [b[2]['Clef-flash'] for b in BENCHMARKS]
    v_kev = [b[2]['Kev 9B'] for b in BENCHMARKS]
    v_ours = [b[2]['FastBrain-9B'] for b in BENCHMARKS]

    ax2.bar(x - w, v_base, width=w, color='#f59e0b', alpha=0.88, label='Clef-flash (9.4B Base · 74.96%)', zorder=3)
    ax2.bar(x, v_kev, width=w, color='#6366f1', alpha=0.88, label='Kev 9B (9B Peer · 76.51%)', zorder=3)
    ax2.bar(x + w, v_ours, width=w, color='#10b981', label='FastBrain-9B (9.4B Ours · 80.55%)', zorder=4)

    ax2.grid(axis='y', linestyle='--', alpha=0.25, color='#8b949e', zorder=0)
    ax2.set_ylim(30, 109)
    ax2.set_xticks(x)
    ax2.set_xticklabels(b_names, rotation=22, ha='right', color='#e6edf3', fontsize=9.2, fontweight='bold')
    ax2.tick_params(axis='y', colors='#8b949e', labelsize=9.5)
    ax2.set_ylabel('Benchmark Score (%)', color='#c9d1d9', fontsize=10.0, fontweight='bold')
    ax2.set_title('B. 9B-Scale Head-to-Head Across All 10 Benchmarks (#1 9B on 10/10)', color='#f0f6fc', fontsize=11.3, fontweight='bold', loc='left', pad=10)

    for i, (vb, vo) in enumerate(zip(v_base, v_ours)):
        delta = vo - vb
        if delta >= 0.5:
            ax2.text(
                x[i] + w, vo + 1.3, f'+{delta:.1f}',
                ha='center', va='bottom', color='#34d399', fontsize=8.2, fontweight='bold'
            )

    leg = ax2.legend(loc='upper right', frameon=True, facecolor='#0d1117', edgecolor='#30363d', fontsize=8.8)
    for text in leg.get_texts():
        text.set_color('#e6edf3')

    out_path = ASSETS / 'benchmark_overview.png'
    fig.savefig(out_path, dpi=240, bbox_inches='tight', pad_inches=0.22, facecolor='#0d1117')
    plt.close(fig)
    return out_path


def build_figure_2_calibration():
    """Figure 2: 3-Panel Fine-Grained Calibration & Reliability Breakdown."""
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(16.5, 5.6), dpi=240)
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.22, left=0.055, right=0.975, top=0.80, bottom=0.14)

    fig.text(0.055, 0.925, 'FastBrain-9B (9.4B) — Fine-Grained Decision Calibration & Open-Set Reliability',
             color='#f0f6fc', fontsize=14.8, fontweight='bold')
    fig.text(0.055, 0.875, 'Same-Harness Comparison vs. Clef-flash (9.4B Base) on Tool Abstention (When2Call), OOS Rejection (CLINC150+OOS), and Security (PhishNChips)',
             color='#8b949e', fontsize=10.0)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#161b22')
    for s in ('top', 'right'):
        ax1.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax1.spines[s].set_color('#30363d')

    cats1 = ['Tool Call\nRecall', 'Cannot Answer\n(Abstain)', 'Request\nClarification', 'Overall\nMacro-F1']
    base1 = [82.86, 40.46, 74.48, 65.58]
    ours1 = [83.32, 80.54, 79.19, 81.13]
    x1 = np.arange(len(cats1))
    w = 0.34
    ax1.bar(x1 - w/2, base1, width=w, color='#f59e0b', alpha=0.85, label='Clef-flash (9.4B Base)', zorder=3)
    ax1.bar(x1 + w/2, ours1, width=w, color='#10b981', label='FastBrain-9B (9.4B)', zorder=4)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, color='#8b949e', zorder=0)
    ax1.set_ylim(25, 98)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cats1, color='#e6edf3', fontsize=8.8, fontweight='bold')
    ax1.tick_params(axis='y', colors='#8b949e', labelsize=9.0)
    ax1.set_ylabel('Recall / Macro-F1 (%)', color='#c9d1d9', fontsize=9.5, fontweight='bold')
    ax1.set_title('A. When2Call (3,652 Requests)', color='#f0f6fc', fontsize=11.0, fontweight='bold', loc='left', pad=8)
    for i, (b, o) in enumerate(zip(base1, ours1)):
        ax1.text(x1[i] - w/2, b + 1.0, f'{b:.1f}%', ha='center', va='bottom', color='#fcd34d', fontsize=7.8)
        ax1.text(x1[i] + w/2, o + 1.0, f'{o:.1f}%', ha='center', va='bottom', color='#34d399', fontsize=8.2, fontweight='bold')
    leg1 = ax1.legend(loc='lower right', frameon=True, facecolor='#0d1117', edgecolor='#30363d', fontsize=8.0)
    for t in leg1.get_texts():
        t.set_color('#e6edf3')

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#161b22')
    for s in ('top', 'right'):
        ax2.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax2.spines[s].set_color('#30363d')

    cats2 = ['Macro-F1\n(151 Classes)', 'Known Intent\nAccuracy', 'OOS Detection\nRecall', 'Known Intent\nRetention (1-FRR)']
    base2 = [66.63, 75.80, 99.10, 77.78]
    ours2 = [98.44, 97.93, 98.50, 98.58]
    x2 = np.arange(len(cats2))
    ax2.bar(x2 - w/2, base2, width=w, color='#f59e0b', alpha=0.85, label='Clef-flash (9.4B Base)', zorder=3)
    ax2.bar(x2 + w/2, ours2, width=w, color='#10b981', label='FastBrain-9B (9.4B)', zorder=4)
    ax2.grid(axis='y', linestyle='--', alpha=0.25, color='#8b949e', zorder=0)
    ax2.set_ylim(55, 106)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats2, color='#e6edf3', fontsize=8.5, fontweight='bold')
    ax2.tick_params(axis='y', colors='#8b949e', labelsize=9.0)
    ax2.set_title('B. CLINC150+OOS (5,500 Requests)', color='#f0f6fc', fontsize=11.0, fontweight='bold', loc='left', pad=8)
    for i, (b, o) in enumerate(zip(base2, ours2)):
        ax2.text(x2[i] - w/2, b + 0.8, f'{b:.1f}%', ha='center', va='bottom', color='#fcd34d', fontsize=7.8)
        ax2.text(x2[i] + w/2, o + 0.8, f'{o:.1f}%', ha='center', va='bottom', color='#34d399', fontsize=8.2, fontweight='bold')

    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('#161b22')
    for s in ('top', 'right'):
        ax3.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax3.spines[s].set_color('#30363d')

    cats3 = ['PhishNChips\nLegit Recall', 'PhishNChips\nPhish Recall', 'PhishNChips\nOverall Acc', 'API-Bank\nAccuracy']
    base3 = [73.30, 76.70, 75.00, 93.11]
    ours3 = [80.50, 77.00, 78.75, 93.70]
    x3 = np.arange(len(cats3))
    ax3.bar(x3 - w/2, base3, width=w, color='#f59e0b', alpha=0.85, label='Clef-flash (9.4B Base)', zorder=3)
    ax3.bar(x3 + w/2, ours3, width=w, color='#10b981', label='FastBrain-9B (9.4B)', zorder=4)
    ax3.grid(axis='y', linestyle='--', alpha=0.25, color='#8b949e', zorder=0)
    ax3.set_ylim(60, 102)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(cats3, color='#e6edf3', fontsize=8.5, fontweight='bold')
    ax3.tick_params(axis='y', colors='#8b949e', labelsize=9.0)
    ax3.set_title('C. PhishNChips & API-Bank', color='#f0f6fc', fontsize=11.0, fontweight='bold', loc='left', pad=8)
    for i, (b, o) in enumerate(zip(base3, ours3)):
        ax3.text(x3[i] - w/2, b + 0.8, f'{b:.1f}%', ha='center', va='bottom', color='#fcd34d', fontsize=7.8)
        ax3.text(x3[i] + w/2, o + 0.8, f'{o:.1f}%', ha='center', va='bottom', color='#34d399', fontsize=8.2, fontweight='bold')

    out_path = ASSETS / 'decision_calibration_breakdown.png'
    fig.savefig(out_path, dpi=240, bbox_inches='tight', pad_inches=0.22, facecolor='#0d1117')
    plt.close(fig)
    return out_path


def build_figure_3_scoreboard():
    """Figure 3: Full 10-Benchmark x 7-Model Scoreboard Matrix with Explicit Parameter Sizes."""
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(17.5, 8.5), dpi=240)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0.1, 17.2)
    ax.set_ylim(0.1, 13.8)
    ax.axis('off')

    ax.text(0.35, 13.20, 'Cloudflare Decision Index (v0.2.1) — 10-Benchmark Evaluation Matrix by Model Size',
            color='#f0f6fc', fontsize=15.0, fontweight='bold')
    ax.text(0.35, 12.72, '22,427 External Evaluation Requests · Parameter Counts Annotated per Model · FastBrain-9B (9.4B) Ranks #2 Overall & #1 Among All <=9B Models',
            color='#8b949e', fontsize=10.0)

    x_cols = [0.35, 4.15, 5.55, 6.95, 8.55, 10.25, 11.65, 13.60, 15.75]
    headers = [
        'Benchmark (Metric)',
        'Clef\n(27.4B)',
        'Clef-flash\n(9.4B Base)',
        'Jev\n(27B)',
        'DiffusionGemma\nJev (26B-A4B)',
        'Kev 9B\n(9B)',
        'Laya\n(0.4B)',
        'FastBrain-9B\n(9.4B · Ours)',
        'vs. Base 9B\n(Clef-flash)',
    ]

    ax.add_patch(FancyBboxPatch(
        (12.40, 0.35), 2.40, 12.00,
        boxstyle='round,pad=0.04,rounding_size=0.10',
        facecolor='#064e3b', edgecolor='#10b981', linewidth=1.6, alpha=0.35, zorder=1
    ))

    y_header = 11.88
    for idx, (xc, h) in enumerate(zip(x_cols, headers)):
        ha = 'left' if idx == 0 else 'center'
        col = '#34d399' if idx == 7 else '#e6edf3'
        ax.text(xc, y_header, h, color=col, fontsize=9.6, fontweight='bold', ha=ha, va='center')

    # Parameter scale badge sub-row
    y_scale = 11.15
    ax.add_patch(FancyBboxPatch(
        (0.30, y_scale - 0.28), 16.55, 0.56,
        boxstyle='round,pad=0.01,rounding_size=0.04',
        facecolor='#1e293b', edgecolor='#334155', linewidth=0.8, alpha=0.85, zorder=0
    ))
    scale_cells = [
        'Backbone / Parameter Scale',
        'Qwen3.8-27B',
        'Qwen3.5-9B',
        '27B Dense',
        '26B (4B MoE)',
        '9B Dense',
        '421M Dense',
        'Qwen3.5-9B',
        'Same 9.4B',
    ]
    for idx, (xc, stxt) in enumerate(zip(x_cols, scale_cells)):
        ha = 'left' if idx == 0 else 'center'
        col = '#6ee7b7' if idx == 7 else '#94a3b8'
        ax.text(xc, y_scale, stxt, color=col, fontsize=8.5, fontweight='bold', ha=ha, va='center')

    ax.plot([0.30, 16.85], [10.72, 10.72], color='#30363d', linewidth=1.3)

    row_y = 10.15
    row_step = 0.88
    peer_keys = ['Clef', 'Clef-flash', 'Jev', 'DiffusionGemma Jev', 'Kev 9B', 'Laya', 'FastBrain-9B']

    for r_idx, (bname, mname, sc, rank_str) in enumerate(BENCHMARKS):
        if r_idx % 2 == 1:
            ax.add_patch(FancyBboxPatch(
                (0.30, row_y - 0.38), 16.55, 0.76,
                boxstyle='round,pad=0.01,rounding_size=0.04',
                facecolor='#161b22', edgecolor='none', alpha=0.75, zorder=0
            ))
        best_v = max(sc[k] for k in peer_keys)
        ax.text(x_cols[0], row_y, f'{bname}  ({mname})', color='#f0f6fc', fontsize=9.5, fontweight='bold', ha='left', va='center')

        for m_i, mk in enumerate(['Clef', 'Clef-flash', 'Jev', 'DiffusionGemma Jev', 'Kev 9B', 'Laya']):
            val = sc[mk]
            is_top = abs(val - best_v) < 1e-6
            ax.text(
                x_cols[1 + m_i], row_y, f'{val:.2f}%',
                color='#ffffff' if is_top else '#c9d1d9',
                fontsize=9.4, fontweight='bold' if is_top else 'normal',
                ha='center', va='center'
            )

        v_ours = sc['FastBrain-9B']
        ax.text(
            x_cols[7], row_y, f'{v_ours:.2f}% ({rank_str})',
            color='#34d399', fontsize=9.6, fontweight='bold', ha='center', va='center'
        )

        delta = v_ours - sc['Clef-flash']
        d_col = '#34d399' if delta > 0.005 else '#9ca3af'
        ax.text(
            x_cols[8], row_y, f'{delta:+.2f} pp',
            color=d_col, fontsize=9.4, fontweight='bold', ha='center', va='center'
        )
        row_y -= row_step

    ax.plot([0.30, 16.85], [row_y + 0.42, row_y + 0.42], color='#30363d', linewidth=1.4)
    y_mean = row_y - 0.12
    ax.add_patch(FancyBboxPatch(
        (0.30, y_mean - 0.42), 16.55, 0.84,
        boxstyle='round,pad=0.02,rounding_size=0.06',
        facecolor='#1f2937', edgecolor='#374151', linewidth=1.1, alpha=0.90, zorder=0
    ))
    ax.text(x_cols[0], y_mean, '10-Benchmark Macro Mean', color='#fbbf24', fontsize=10.1, fontweight='bold', ha='left', va='center')
    for m_i, mk in enumerate(['Clef', 'Clef-flash', 'Jev', 'DiffusionGemma Jev', 'Kev 9B', 'Laya']):
        ax.text(x_cols[1 + m_i], y_mean, f'{MACRO_MEANS[mk]:.2f}%', color='#f0f6fc', fontsize=9.7, fontweight='bold', ha='center', va='center')

    ax.text(x_cols[7], y_mean, f"{MACRO_MEANS['FastBrain-9B']:.2f}% (#2 / #1 9B)", color='#34d399', fontsize=9.9, fontweight='bold', ha='center', va='center')
    d_mean = MACRO_MEANS['FastBrain-9B'] - MACRO_MEANS['Clef-flash']
    ax.text(x_cols[8], y_mean, f'{d_mean:+.2f} pp', color='#34d399', fontsize=9.9, fontweight='bold', ha='center', va='center')

    out_path = ASSETS / 'benchmark_scoreboard.png'
    fig.savefig(out_path, dpi=240, bbox_inches='tight', pad_inches=0.22, facecolor='#0d1117')
    plt.close(fig)
    return out_path


def build_readme() -> Path:
    readme_text = '''---
base_model: Cloudflare/clef-flash
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
  - fastbrain
  - systemone
  - decision-index
  - tool-calling
  - intent-classification
  - structured-decision
---

<div align="center">

# FastBrain-9B

**Non-Autoregressive System-1 Structured Decision Model (`9.4B` Parameters)**

[![Architecture](https://img.shields.io/badge/Backbone-Qwen3.5--9B_Hybrid_(9.4B_Total)-3b82f6?style=flat-square)](#architecture)
[![Decision Head](https://img.shields.io/badge/Head-JointSchemaHead_(116M)-10b981?style=flat-square)](#architecture)
[![Decision Index](https://img.shields.io/badge/Decision_Index_v0.2.1-80.55%25_(%232_Overall_%C2%B7_%231_9B)-f59e0b?style=flat-square)](#benchmark-evaluation)
[![License](https://img.shields.io/badge/License-Apache_2.0-6366f1?style=flat-square)](./LICENSE)

</div>

**FastBrain-9B** (`9.4B` total parameters: `Qwen3.5-9B` hybrid backbone + `116M` `JointSchemaHead`) is a single-forward-pass, non-autoregressive **System-1 structured decision model**.

Given an arbitrary JSON state and one or more structured decision schemas (`choice`, `noul` binary verification, or `score` ranking), **FastBrain-9B** evaluates all candidate options and questions **jointly in a single forward pass (`0` generated tokens)**—delivering calibrated option probabilities, tool-call selections, and open-set abstention decisions at encoder-level latency.

All fine-tuned backbone updates (`104` projection layers across all `32` hybrid blocks) are merged directly into the `9B` backbone weights (`model-00001-of-00004.safetensors` .. `model-00004-of-00004.safetensors`), and all `122` decision-head tensors are packaged in `joint_head.safetensors`. **No external LoRA or PEFT adapter is required at inference time.**

---

## Benchmark Evaluation & Parameter Scale Comparison

Evaluated on the official **Cloudflare Decision Index (v0.2.1)** across **10 benchmarks (22,427 evaluation requests)**, **FastBrain-9B (`9.4B`)** achieves an overall Macro Mean of **`80.55%`**:
- **Ranks `#1` across all `<=9B` models on `10 / 10` benchmarks**, outperforming the same-size base model `Clef-flash` (`9.4B`, `74.96%`) by **`+5.59 pp`**, `Kev 9B` (`9B`, `76.51%`) by **`+4.04 pp`**, and `Laya` (`0.4B`, `58.09%`) by **`+22.46 pp`**.
- **Outperforms `3x`-larger `27B`-class models**: surpasses `Jev` (`27B`, `80.32%`) and `DiffusionGemma Jev` (`26B-A4B`, `69.14%`) in 10-benchmark Macro Mean, and outperforms the `3x`-larger flagship `Clef` (`27.4B`, `81.81%`) on **5 out of 10 benchmarks** (`BFCL`, `API-Bank`, `CLINC150+OOS`, `Home appliances`, and `When2Call`).

![FastBrain-9B Overview](./assets/benchmark_overview.png)

### Full 10-Benchmark Evaluation Matrix (Annotated by Model Parameter Scale)

![FastBrain-9B Scoreboard](./assets/benchmark_scoreboard.png)

### Fine-Grained Decision Calibration & Open-Set Reliability

![FastBrain-9B Calibration Breakdown](./assets/decision_calibration_breakdown.png)

### Summary Comparison by Model Parameter Scale

| Benchmark | Metric | `Laya`<br>*(0.4B / 421M)* | `DiffusionGemma Jev`<br>*(26B-A4B · 4B Act)* | `Clef-flash`<br>*(9.4B Base)* | `Kev 9B`<br>*(9B Dense)* | `Jev`<br>*(27B Dense)* | `Clef`<br>*(27.4B Flagship)* | **FastBrain-9B**<br>*(9.4B · Ours)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parameter Scale** | **Total (Active)** | `0.42B` | `25.8B (4B MoE)` | `9.41B` | `9.0B` | `27.0B` | `27.36B` | **`9.41B`** |
| **BFCL** | Case Exact | 95.87% | 97.64% | 98.76% | 97.58% | 98.58% | 98.47% | **98.76% (#1)** |
| **ToolRet** | nDCG@10 | 51.31% | 45.28% | 67.26% | 65.72% | 65.50% | 68.18% | **67.26% (#2 · #1 9B)** |
| **API-Bank** | Accuracy | 86.22% | 85.24% | 93.11% | 89.76% | 90.35% | 91.93% | **93.70% (#1)** |
| **BANKING77** | Macro-F1 | 62.66% | 93.80% | 93.69% | 91.77% | 93.58% | 94.07% | **93.77% (#1 9B)** |
| **CLINC150+OOS** | Macro-F1 | 73.79% | 94.64% | 66.77% | 96.65% | 96.44% | 97.43% | **98.44% (#1)** |
| **Home appliances** | Case Exact | 23.86% | 68.18% | 96.59% | 88.64% | 88.64% | 96.59% | **97.73% (#1)** |
| **BRIGHT** | nDCG@10 | 25.00% | 18.69% | 38.49% | 38.53% | 43.37% | 48.03% | **38.58% (#1 9B)** |
| **Amazon ESCI** | Macro-F1 | 52.84% | 56.83% | 57.39% | 53.75% | 56.60% | 59.30% | **57.40% (#2 · #1 9B)** |
| **PhishNChips** | Accuracy | 60.45% | 55.65% | 71.95% | 72.10% | 88.20% | 91.70% | **78.75% (#1 9B)** |
| **When2Call** | Macro-F1 | 48.92% | 75.44% | 65.58% | 70.64% | 81.91% | 72.37% | **81.13% (#2 · #1 9B)** |
| **Macro Mean (10)** | **Average** | **58.09%** | **69.14%** | **74.96%** | **76.51%** | **80.32%** | **81.81%** | **80.55% (#2 · #1 9B)** |

---

## Architecture

1. **Hybrid Gated DeltaNet + Full-Attention Backbone (`9.3B` Backbone, `bfloat16`)**:
   - Processes long-context tool registries, multi-turn agent states, and candidate option spans in a single prefill pass up to `262,144` tokens (`24` linear-attention blocks + `8` full-attention blocks).
   - Jointly adapted across all `104` linear-attention (`in_proj_qkv`, `in_proj_z`, `out_proj`) and self-attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) projections and merged in-place into the backbone weights.
2. **Joint Schema Decision Head (`joint_head.safetensors`, `116M` Parameters)**:
   - Extracts question-level and option-level span representations directly from the final hidden states and scores all questions (`choice`, `noul`, `score`) simultaneously without autoregressive decoding (`output_tokens = 0`).
3. **Compile-Time AST Schema-Aware Margin Calibration (`joint_schema_model.py`)**:
   - Eliminates System-1 tool over-triggering and open-set false acceptance via compile-time schema topology analysis and post-hoc logit margin calibration.

---

## Quickstart

`TypeScript/FastBrain-9B` is a drop-in replacement for `Cloudflare/clef-flash` using the standard `joint_schema_model` runtime:

```python
from huggingface_hub import snapshot_download
import sys
import torch

model_dir = snapshot_download("TypeScript/FastBrain-9B")
sys.path.insert(0, model_dir)
import joint_schema_model as clef

model, processor = clef.load_release_model(model_dir, device="cuda", dtype=torch.bfloat16)

response = clef.systemone(model, processor, {
    "model": "TypeScript/FastBrain-9B",
    "state": {
        "query": "Can you check my account balance?",
        "tools": ["get_balance", "transfer_funds"],
    },
    "questions": {
        "q": {
            "type": "choice",
            "query": "Select the best next action.",
            "criteria": {
                "a": "Call <tool_call>{\\"name\\": \\"get_balance\\", \\"arguments\\": {}}</tool_call>",
                "b": "Could you please provide your account ID or authentication token first?",
                "c": "I am unable to access personal banking records.",
            },
        },
    },
})
print(response)
```

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `model-00001-of-00004.safetensors` .. `00004` | Full `9B` backbone weights (`bfloat16`, in-place merged, no LoRA required) |
| `joint_head.safetensors` & `joint_head_config.json` | `116M`-parameter `JointSchemaHead` weights (`122` tensors) and configuration |
| `joint_schema_model.py` | `ClefModel`, `JointSchemaHead`, `load_release_model()`, and `systemone()` runtime |
| `config.json`, `tokenizer.json`, `processor_config.json` | Standard Hugging Face model, tokenizer, and processor configurations |
| `assets/*.png` | High-resolution technical benchmark and calibration charts |
'''
    out = ASSETS / 'README.md'
    out.write_text(readme_text)
    return out


def main():
    p1 = build_figure_1_hero()
    p2 = build_figure_2_calibration()
    p3 = build_figure_3_scoreboard()
    pr = build_readme()
    print('Generated:', p1, p2, p3, pr)


if __name__ == '__main__':
    main()
