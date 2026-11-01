"""Generate 6 publication-grade matplotlib figures (vector PDF + 300-DPI PNG) for the FastBrain-9B arXiv paper."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

PAPER_ROOT = Path('/data00/home/xiangyang.6/multica_workspaces/1aa199be-a815-4eac-aed2-7fcd7948963b/08c414e0/workdir/scratch/fastbrain_arxiv_paper_20261029')
FIG_DIR = PAPER_ROOT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent academic publication style with larger typography (>=8pt after 2-column scaling)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#334155',
    'axes.linewidth': 1.0,
    'grid.color': '#cbd5e1',
    'grid.linestyle': '--',
    'grid.alpha': 0.55,
    'xtick.color': '#1e293b',
    'ytick.color': '#1e293b',
    'text.color': '#0f172a',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

PALETTE = {
    'ours': '#059669',       # Emerald green
    'ours_light': '#d1fae5',
    'clef27': '#2563eb',     # Royal blue
    'jev27': '#4f46e5',      # Indigo
    'kev9': '#64748b',       # Slate
    'base9': '#d97706',      # Amber/Orange
    'diff26': '#7c3aed',     # Violet
    'laya': '#94a3b8',       # Light slate
    'red': '#dc2626',        # Crimson
    'dark': '#0f172a',
}


def save_both(fig, stem: str):
    pdf_path = FIG_DIR / f'{stem}.pdf'
    png_path = FIG_DIR / f'{stem}.png'
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.08)
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return pdf_path, png_path


def build_fig1_architecture():
    """Fig 1: End-to-End Architecture & Three-Stage Calibration Pipeline of FastBrain-9B."""
    fig, ax = plt.subplots(figsize=(15.2, 5.4), dpi=300)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0.1, 5.4)
    ax.axis('off')

    # Box 1: Structured Request Input
    ax.add_patch(FancyBboxPatch((0.20, 1.05), 2.85, 4.15, boxstyle='round,pad=0.08,rounding_size=0.15',
                                facecolor='#f8fafc', edgecolor='#64748b', linewidth=1.4))
    ax.text(1.625, 4.88, '1. Structured Request', fontsize=11.2, fontweight='bold', ha='center', color='#0f172a')
    ax.text(1.625, 4.56, 'State S + Questions {Q_k}', fontsize=9.8, ha='center', color='#475569')

    code_box = (
        '{\n'
        '  "state": {\n'
        '    "query": "...",\n'
        '    "tools": [53 APIs]\n'
        '  },\n'
        '  "questions": {\n'
        '    "q1": {"type":"choice"},\n'
        '    "q2": {"type":"noul"}\n'
        '  }\n'
    ) + '}'
    ax.add_patch(FancyBboxPatch((0.36, 2.20), 2.53, 2.12, boxstyle='round,pad=0.05,rounding_size=0.08',
                                facecolor='#f1f5f9', edgecolor='#cbd5e1', linewidth=1.0))
    ax.text(0.45, 3.26, code_box, fontsize=8.1, family='monospace', va='center', ha='left', color='#1e293b')
    ax.text(1.625, 1.56, 'Span-Indexed Prefill\nQuestion & Option Spans\n16,384 Eval / 262k Max Tok',
            fontsize=8.6, ha='center', va='center', color='#334155', fontweight='bold')

    # Arrow 1 -> 2
    ax.annotate('', xy=(3.40, 3.12), xytext=(3.07, 3.12),
                arrowprops=dict(arrowstyle='->', lw=2.0, color='#334155'))

    # Box 2: Hybrid Qwen3.5-9B Backbone (In-Place Merged)
    ax.add_patch(FancyBboxPatch((3.42, 1.05), 3.65, 4.15, boxstyle='round,pad=0.08,rounding_size=0.15',
                                facecolor='#eff6ff', edgecolor='#2563eb', linewidth=1.5))
    ax.text(5.245, 4.88, '2. Hybrid 9B Backbone (32L)', fontsize=10.6, fontweight='bold', ha='center', color='#1e3a8a')
    ax.text(5.245, 4.56, 'Qwen3.5-9B · 4 bfloat16 Shards', fontsize=9.2, ha='center', color='#1d4ed8')

    ax.add_patch(FancyBboxPatch((3.62, 3.30), 3.25, 1.06, boxstyle='round,pad=0.05,rounding_size=0.10',
                                facecolor='#dbeafe', edgecolor='#3b82f6', linewidth=1.1))
    ax.text(5.245, 4.00, '24× Gated DeltaNet Blocks', fontsize=9.4, fontweight='bold', ha='center', color='#1e3a8a')
    ax.text(5.245, 3.58, 'O(L) State Recurrence · 72 Projections\nin_proj_qkv, in_proj_z, out_proj',
            fontsize=8.3, ha='center', color='#1e40af')

    ax.add_patch(FancyBboxPatch((3.62, 2.02), 3.25, 1.06, boxstyle='round,pad=0.05,rounding_size=0.10',
                                facecolor='#e0e7ff', edgecolor='#4f46e5', linewidth=1.1))
    ax.text(5.245, 2.72, '8× Full Self-Attention Blocks', fontsize=9.4, fontweight='bold', ha='center', color='#312e81')
    ax.text(5.245, 2.30, 'Cross-Span Routing · 32 Projections\nq_proj, k_proj, v_proj, o_proj',
            fontsize=8.3, ha='center', color='#3730a3')

    ax.text(5.245, 1.48, 'Adapter Merge (λ=0.50, 208 A/B Tensors):\nW_merged = W_0 + (α/r) B_fused A_fused',
            fontsize=8.0, ha='center', va='center', color='#0f172a', fontweight='bold')

    # Arrow 2 -> 3
    ax.annotate('', xy=(7.42, 3.12), xytext=(7.09, 3.12),
                arrowprops=dict(arrowstyle='->', lw=2.0, color='#334155'))

    # Box 3: JointSchemaHead + Selective Dual-Teacher Training
    ax.add_patch(FancyBboxPatch((7.44, 1.05), 3.70, 4.15, boxstyle='round,pad=0.08,rounding_size=0.15',
                                facecolor='#ecfdf5', edgecolor='#059669', linewidth=1.5))
    ax.text(9.29, 4.88, '3. JointSchemaHead (116.1M)', fontsize=10.6, fontweight='bold', ha='center', color='#065f46')
    ax.text(9.29, 4.56, 'joint_head.safetensors (122 Tensors)', fontsize=9.2, ha='center', color='#047857')

    ax.add_patch(FancyBboxPatch((7.64, 3.30), 3.30, 1.06, boxstyle='round,pad=0.05,rounding_size=0.10',
                                facecolor='#d1fae5', edgecolor='#10b981', linewidth=1.1))
    ax.text(9.29, 4.00, 'Bilinear Question–Option Readout', fontsize=9.3, fontweight='bold', ha='center', color='#065f46')
    ax.text(9.29, 3.58, 'choice: M_k Option Softmax / Ranking\nnoul: Binary Verification · score: Ranking',
            fontsize=8.3, ha='center', color='#064e3b')

    ax.add_patch(FancyBboxPatch((7.64, 2.02), 3.30, 1.06, boxstyle='round,pad=0.05,rounding_size=0.10',
                                facecolor='#fef3c7', edgecolor='#d97706', linewidth=1.1))
    ax.text(9.29, 2.72, 'Dual-Teacher KL + Margin Loss', fontsize=9.3, fontweight='bold', ha='center', color='#92400e')
    ax.text(9.29, 2.30, 'Anchor Teacher: Tool/Ranking Tasks\nGovernance Teacher: CLINC, W2C, Phish',
            fontsize=8.3, ha='center', color='#78350f')

    ax.text(9.29, 1.48, 'L = L_CE + β_B L_Brier + β_m Hinge\n+ α_KL T_d² KL(p_teach || p_θ)',
            fontsize=8.1, ha='center', va='center', color='#0f172a', fontweight='bold')

    # Arrow 3 -> 4
    ax.annotate('', xy=(11.49, 3.12), xytext=(11.16, 3.12),
                arrowprops=dict(arrowstyle='->', lw=2.0, color='#334155'))

    # Box 4: Request-Time Schema-Dependent Post-Processing
    ax.add_patch(FancyBboxPatch((11.51, 1.05), 3.49, 4.15, boxstyle='round,pad=0.08,rounding_size=0.15',
                                facecolor='#f5f3ff', edgecolor='#7c3aed', linewidth=1.5))
    ax.text(13.255, 4.88, '4. Schema Post-Processing', fontsize=10.6, fontweight='bold', ha='center', color='#4c1d95')
    ax.text(13.255, 4.56, 'Request-Time Topology T(S, Q)', fontsize=9.2, ha='center', color='#5b21b6')

    ax.add_patch(FancyBboxPatch((11.69, 3.30), 3.13, 1.06, boxstyle='round,pad=0.05,rounding_size=0.10',
                                facecolor='#ede9fe', edgecolor='#8b5cf6', linewidth=1.1))
    ax.text(13.255, 4.00, 'Role-Conditioned Offsets', fontsize=9.4, fontweight='bold', ha='center', color='#4c1d95')
    ax.text(13.255, 3.58, 'cannot_answer: b = +0.34\ninfo / tool_call: b = +0.18',
            fontsize=8.3, ha='center', color='#5b21b6')

    ax.add_patch(FancyBboxPatch((11.69, 2.02), 3.13, 1.06, boxstyle='round,pad=0.05,rounding_size=0.10',
                                facecolor='#fae8ff', edgecolor='#a855f7', linewidth=1.1))
    ax.text(13.255, 2.72, 'Two-Probe Verdict & Routing', fontsize=9.3, fontweight='bold', ha='center', color='#581c87')
    ax.text(13.255, 2.30, 'Two-Probe Mean (γ = 0.60)\nDual-Potential OOS & Deferral',
            fontsize=8.3, ha='center', color='#6b21a8')

    ax.text(13.255, 1.48, '80.55% Unweighted 10-Task Mean\n(#1 9B-Class on 10/10 · >27B Jev)',
            fontsize=8.1, ha='center', va='center', color='#059669', fontweight='bold')

    # Bottom summary banner
    ax.add_patch(FancyBboxPatch((0.20, 0.18), 14.80, 0.66, boxstyle='round,pad=0.05,rounding_size=0.10',
                                facecolor='#f1f5f9', edgecolor='#94a3b8', linewidth=1.1))
    ax.text(7.60, 0.51,
            'Single-Pass Deployment: 104 adapted backbone projections are merged into 4 bfloat16 shards + joint_head.safetensors '
            '(1 prefill pass, 0 generated tokens, 24.71 GB max observed GPU memory on RTX PRO 6000 Blackwell).',
            fontsize=8.6, ha='center', va='center', color='#1e293b', fontweight='bold')

    return save_both(fig, 'fig1_architecture_pipeline')


def build_fig2_pareto_and_peers():
    """Fig 2: Parameter Scale vs. Unweighted 10-Benchmark Mean + Per-Benchmark Comparison."""
    fig = plt.figure(figsize=(15.4, 5.2), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.02, 1.33], wspace=0.23, left=0.06, right=0.985, top=0.90, bottom=0.20)

    # Panel A: Parameter Scale vs 10-Benchmark Mean Scatter + Pareto Curve
    ax1 = fig.add_subplot(gs[0])
    models_scatter = [
        ('Laya\n(0.42B)', 0.42, 58.09, PALETTE['laya'], 'o', (10, -8)),
        ('Clef-flash\n(9.41B Base)', 9.41, 74.96, PALETTE['base9'], 's', (10, -18)),
        ('Kev 9B\n(9.0B)', 9.00, 76.51, PALETTE['kev9'], 'D', (-58, -6)),
        ('FastBrain-9B\n(9.41B · Ours)', 9.41, 80.55, PALETTE['ours'], '*', (-20, 10)),
        ('DiffusionGemma Jev\n(25.8B / 4B MoE)', 25.82, 69.14, PALETTE['diff26'], '^', (-95, -20)),
        ('Jev\n(27.0B)', 27.00, 80.32, PALETTE['jev27'], 'v', (-46, -20)),
        ('Clef\n(27.36B)', 27.36, 81.81, PALETTE['clef27'], 'P', (-60, 8)),
    ]

    pareto_x = [0.42, 9.41, 27.36]
    pareto_y = [58.09, 80.55, 81.81]
    ax1.plot(pareto_x, pareto_y, linestyle='--', linewidth=2.0, color='#059669', alpha=0.75, label='Empirical Pareto Frontier', zorder=2)

    ax1.annotate('', xy=(9.41, 80.15), xytext=(9.41, 75.40),
                 arrowprops=dict(arrowstyle='->', lw=2.2, color='#059669'))
    ax1.text(10.2, 77.7, '+5.59 pp Pub.\n(+5.66 pp Same-Harness)', fontsize=9.2, fontweight='bold', color='#059669', va='center')

    for label, x_p, y_m, col, marker, offset in models_scatter:
        sz = 250 if 'FastBrain' in label else 130
        ax1.scatter([x_p], [y_m], s=sz, color=col, marker=marker, edgecolor='white', linewidth=1.3, zorder=4)
        ax1.annotate(
            f'{label}\n({y_m:.2f}%)',
            xy=(x_p, y_m), xytext=offset, textcoords='offset points',
            fontsize=9.0, fontweight='bold' if 'FastBrain' in label else 'normal',
            color='#065f46' if 'FastBrain' in label else '#1e293b',
        )

    ax1.grid(True, zorder=0)
    ax1.set_xlim(-1.5, 31.5)
    ax1.set_ylim(53.5, 87.0)
    ax1.tick_params(axis='both', labelsize=9.5)
    ax1.set_xlabel('Total Model Parameters (Billions)', fontsize=10.5, fontweight='bold')
    ax1.set_ylabel('Unweighted 10-Benchmark Mean (%)', fontsize=10.5, fontweight='bold')
    ax1.set_title('a  10-Benchmark Mean vs. Parameter Scale', fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax1.legend(loc='lower right', frameon=True, fontsize=9.2)

    # Panel B: 10-Benchmark Grouped Bar Comparison
    ax2 = fig.add_subplot(gs[1])
    b_names = ['BFCL', 'ToolRet', 'API-Bank', 'BANKING77', 'CLINC150+OOS', 'Home Appl.', 'BRIGHT', 'Amazon ESCI', 'PhishNChips', 'When2Call']
    v_flash = [98.76, 67.26, 93.11, 93.69, 66.77, 96.59, 38.49, 57.39, 71.95, 65.58]
    v_jev27 = [98.58, 65.50, 90.35, 93.58, 96.44, 88.64, 43.37, 56.60, 88.20, 81.91]
    v_clef27 = [98.47, 68.18, 91.93, 94.07, 97.43, 96.59, 48.03, 59.30, 91.70, 72.37]
    v_ours = [98.76, 67.26, 93.70, 93.77, 98.44, 97.73, 38.58, 57.40, 78.75, 81.13]

    x = np.arange(len(b_names))
    w = 0.20
    ax2.bar(x - 1.5*w, v_flash, width=w, color=PALETTE['base9'], label='Clef-flash (9.41B · 74.96%)', zorder=3)
    ax2.bar(x - 0.5*w, v_jev27, width=w, color=PALETTE['jev27'], label='Jev (27B · 80.32%)', zorder=3)
    ax2.bar(x + 0.5*w, v_clef27, width=w, color=PALETTE['clef27'], label='Clef (27.36B · 81.81%)', zorder=3)
    ax2.bar(x + 1.5*w, v_ours, width=w, color=PALETTE['ours'], label='FastBrain-9B (9.41B · 80.55%)', zorder=4)

    for i, (vf, vc, vo) in enumerate(zip(v_flash, v_clef27, v_ours)):
        d = vo - vf
        if d >= 0.5:
            if i == 8:
                ax2.annotate(f'+{d:.1f}', xy=(x[i] + 1.5*w, vo), xytext=(x[i] + 1.85*w, vo + 2.5),
                             ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#059669',
                             bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.9), zorder=6)
            else:
                ax2.text(x[i] + 1.5*w, max(vo, vc) + 1.2, f'+{d:.1f}', ha='center', va='bottom',
                         fontsize=8.8, fontweight='bold', color='#059669', zorder=6)

    ax2.grid(axis='y', zorder=0)
    ax2.set_ylim(30, 117)
    ax2.set_xticks(x)
    ax2.set_xticklabels(b_names, rotation=24, ha='right', fontsize=9.4, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=9.5)
    ax2.set_ylabel('Benchmark Score (%)', fontsize=10.5, fontweight='bold')
    ax2.set_title('b  Per-Benchmark Scores Across 10 Decision Index Tasks', fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax2.legend(loc='upper right', frameon=True, fontsize=8.8, ncol=2)

    return save_both(fig, 'fig2_overall_pareto_and_peers')


def build_fig3_governance_diagnostics():
    """Fig 3: 3-Panel Governance & Boundary Calibration Analysis (CLINC150+OOS, When2Call, PhishNChips)."""
    fig = plt.figure(figsize=(15.6, 5.0), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.05, 1.18, 1.0], wspace=0.25, left=0.055, right=0.985, top=0.83, bottom=0.18)

    # Panel A: CLINC150+OOS Known-Intent Retention vs OOS Recall (exact round11 vs round14)
    ax1 = fig.add_subplot(gs[0])
    metrics_clinc = ['151-Class\nMacro-F1', 'Known Intent\nAccuracy', 'Known Intent\nRetention (1-FRR)', 'OOS Rejection\nRecall']
    base_c = [66.63, 55.24, 60.18, 99.00]
    merge_c = [97.96, 97.36, 98.20, 97.90]
    ours_c = [98.44, 97.93, 98.58, 98.50]

    x1 = np.arange(len(metrics_clinc))
    w = 0.25
    ax1.bar(x1 - w, base_c, width=w, color=PALETTE['base9'], label='Clef-flash (Base 9.4B)', zorder=3)
    ax1.bar(x1, merge_c, width=w, color=PALETTE['clef27'], label='Adapter-Head Merged', zorder=3)
    ax1.bar(x1 + w, ours_c, width=w, color=PALETTE['ours'], label='FastBrain-9B (Full)', zorder=4)

    for i, (b, o) in enumerate(zip(base_c, ours_c)):
        ax1.text(x1[i] - 1.08*w, b + 1.0, f'{b:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#92400e')
        ax1.text(x1[i] + 1.08*w, o + 1.0, f'{o:.1f}', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#065f46')

    ax1.grid(axis='y', zorder=0)
    ax1.set_ylim(40, 118)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(metrics_clinc, fontsize=9.0, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=9.2)
    ax1.set_ylabel('Score / Rate (%)', fontsize=10.2, fontweight='bold')
    ax1.set_title('a  CLINC150+OOS Boundary Calibration', fontsize=11.2, fontweight='bold', loc='left', pad=30)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=True, fontsize=8.0, columnspacing=0.7)

    # Panel B: When2Call Error Mode Reduction (Exact alignment with Table 3 round11 & round14)
    ax2 = fig.add_subplot(gs[1])
    w2c_cats = [
        'Tool Call\nRecall',
        'Cannot Answer\nRecall',
        'Clarification\nRecall',
        'False Direct\nError ↓',
        'Premature Tool\nError ↓',
    ]
    base_w = [82.86, 40.46, 74.48, 33.82, 19.87]
    fuse_w = [81.70, 70.81, 79.10, 17.07, 14.97]
    ours_w = [83.32, 80.54, 79.19, 9.96, 15.07]

    x2 = np.arange(len(w2c_cats))
    ax2.bar(x2 - w, base_w, width=w, color=PALETTE['base9'], label='Clef-flash (Base)', zorder=3)
    ax2.bar(x2, fuse_w, width=w, color=PALETTE['clef27'], label='Adapter-Head Merged', zorder=3)
    ax2.bar(x2 + w, ours_w, width=w, color=PALETTE['ours'], label='FastBrain-9B (Full)', zorder=4)

    for i, (b, o) in enumerate(zip(base_w, ours_w)):
        ax2.text(x2[i] - 1.10*w, b + 1.0, f'{b:.1f}', ha='center', va='bottom', fontsize=8.3, fontweight='bold', color='#92400e')
        ax2.text(x2[i] + 1.10*w, o + 1.0, f'{o:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#065f46')

    ax2.grid(axis='y', zorder=0)
    ax2.set_ylim(0, 98)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(w2c_cats, fontsize=8.8, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=9.2)
    ax2.set_ylabel('Recall / Error Rate (%)', fontsize=10.2, fontweight='bold')
    ax2.set_title('b  When2Call Action-Role Calibration', fontsize=11.2, fontweight='bold', loc='left', pad=30)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=True, fontsize=8.0, columnspacing=0.7)

    # Panel C: PhishNChips Legitimate vs Phishing Recall Balance & Two-Probe Average
    ax3 = fig.add_subplot(gs[2])
    phish_cats = ['Legitimate\nMail Recall', 'Phishing\nMail Recall', 'Overall\nAccuracy', 'Recall\nGap |Δ| ↓']
    base_p = [73.30, 76.70, 75.00, 3.40]
    unbal_p = [70.40, 82.40, 76.40, 12.00]
    ours_p = [80.50, 77.00, 78.75, 3.50]

    x3 = np.arange(len(phish_cats))
    ax3.bar(x3 - w, base_p, width=w, color=PALETTE['base9'], label='Clef-flash (Base)', zorder=3)
    ax3.bar(x3, unbal_p, width=w, color=PALETTE['red'], alpha=0.82, label='Pairwise Margin SFT', zorder=3)
    ax3.bar(x3 + w, ours_p, width=w, color=PALETTE['ours'], label='FastBrain-9B (Full)', zorder=4)

    for i, (b, o) in enumerate(zip(base_p, ours_p)):
        ax3.text(x3[i] - 1.05*w, b + 1.0, f'{b:.1f}', ha='center', va='bottom', fontsize=8.4, fontweight='bold', color='#92400e')
        ax3.text(x3[i] + 1.05*w, o + 1.0, f'{o:.1f}', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#065f46')

    ax3.grid(axis='y', zorder=0)
    ax3.set_ylim(0, 98)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(phish_cats, fontsize=8.8, fontweight='bold')
    ax3.tick_params(axis='y', labelsize=9.2)
    ax3.set_ylabel('Recall / Accuracy (%)', fontsize=10.2, fontweight='bold')
    ax3.set_title('c  PhishNChips Two-Probe Verdict Average', fontsize=11.2, fontweight='bold', loc='left', pad=30)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=True, fontsize=8.0, columnspacing=0.7)

    return save_both(fig, 'fig3_governance_calibration_diagnostics')


def build_fig4_ablations():
    """Fig 4: Component Ablation Progression, Adapter-Head Interpolation, and Univariate Offset Sweep."""
    fig = plt.figure(figsize=(15.6, 5.0), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 0.98, 1.02], wspace=0.25, left=0.055, right=0.985, top=0.88, bottom=0.20)

    # Panel A: Stepwise Component Ablation across 6 Stages (Exact match with Table 3)
    ax1 = fig.add_subplot(gs[0])
    stages = ['Base\n(9.4B)', '+Margin\nSFT', '+Dual\nTeacher', '+Adapter\nMerge', '+Role &\n2-Probe', 'Full\nFastBrain']
    s_mean = [74.89, 79.06, 79.19, 79.31, 80.08, 80.55]
    s_apibank = [93.11, 91.34, 93.11, 92.72, 92.72, 93.70]
    s_w2c = [65.39, 75.63, 77.00, 77.08, 81.13, 81.13]
    s_phish = [75.00, 76.40, 74.75, 75.15, 78.75, 78.75]

    x1 = np.arange(len(stages))
    ax1.plot(x1, s_apibank, marker='s', linewidth=2.1, color=PALETTE['clef27'], label='API-Bank (53-Tool Selection)')
    ax1.plot(x1, s_w2c, marker='^', linewidth=2.1, color=PALETTE['base9'], label='When2Call (4-Way Governance)')
    ax1.plot(x1, s_phish, marker='D', linewidth=1.9, color=PALETTE['diff26'], label='PhishNChips (9-Probe Security)')
    ax1.plot(x1, s_mean, marker='o', linewidth=2.7, color=PALETTE['ours'], label='10-Benchmark Mean')

    for i, m_val in enumerate(s_mean):
        y_off = -15 if i == 0 else (8 if i != 4 else -19)
        x_off = 10 if i == 0 else (-16 if i == 4 else (8 if i == 5 else 0))
        ax1.annotate(f'{m_val:.2f}%', (x1[i], m_val), xytext=(x_off, y_off), textcoords='offset points',
                     ha='center', fontsize=8.5, fontweight='bold', color='#065f46')

    ax1.grid(True, zorder=0)
    ax1.set_ylim(62, 98.5)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(stages, fontsize=8.8, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=9.2)
    ax1.set_ylabel('Benchmark Score (%)', fontsize=10.2, fontweight='bold')
    ax1.set_title('a  Stepwise Component Ablation Trajectory', fontsize=11.2, fontweight='bold', loc='left', pad=8)
    ax1.legend(loc='lower right', frameon=True, fontsize=8.4)

    # Panel B: Synchronized Adapter-Factor + JointSchemaHead Interpolation (Exact optimum lambda=0.50)
    ax2 = fig.add_subplot(gs[1])
    lambdas = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    sync_merge = np.array([79.19, 79.27, 79.31, 79.22, 79.06])
    head_only = np.array([79.19, 78.72, 78.38, 78.60, 79.06])
    lora_only = np.array([79.19, 78.48, 77.95, 78.25, 79.06])

    ax2.plot(lambdas, sync_merge, marker='o', linewidth=2.5, color=PALETTE['ours'],
             label='Synchronized LoRA + Head')
    ax2.plot(lambdas, head_only, marker='s', linestyle='--', linewidth=1.9, color=PALETTE['clef27'],
             label='JointSchemaHead-Only Merge')
    ax2.plot(lambdas, lora_only, marker='^', linestyle=':', linewidth=1.9, color=PALETTE['red'],
             label='Backbone LoRA-Only Merge')

    ax2.scatter([0.50], [79.31], s=140, color=PALETTE['ours'], edgecolor='black', zorder=5)
    ax2.annotate('Selected λ=0.50\n(79.31% Mean)', xy=(0.50, 79.31), xytext=(0.30, 79.40),
                 fontsize=9.0, fontweight='bold', color='#065f46')

    ax2.grid(True, zorder=0)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(77.6, 79.65)
    ax2.tick_params(axis='both', labelsize=9.2)
    ax2.set_xlabel('Interpolation Weight λ (Anchor vs. Boundary)', fontsize=9.8, fontweight='bold')
    ax2.set_ylabel('10-Benchmark Mean (%)', fontsize=10.2, fontweight='bold')
    ax2.set_title('b  Co-Adapted LoRA & Head Interpolation', fontsize=11.2, fontweight='bold', loc='left', pad=8)
    ax2.legend(loc='lower center', frameon=True, fontsize=8.4)

    # Panel C: True Univariate Sweep of b_cannot on When2Call (holding b_tool=0.18, b_info=0.18 fixed)
    ax3 = fig.add_subplot(gs[2])
    b_vals = np.array([0.00, 0.10, 0.20, 0.28, 0.34, 0.40, 0.48])
    w2c_acc = np.array([77.05, 78.04, 79.22, 80.64, 81.13, 81.30, 82.23])
    cannot_rec = np.array([67.49, 70.50, 74.13, 78.46, 80.54, 82.08, 86.49])
    tool_rec = np.array([84.32, 84.25, 84.17, 84.02, 83.32, 82.86, 81.62])
    info_rec = np.array([79.85, 79.66, 79.38, 79.19, 79.19, 78.44, 77.78])

    ax3.plot(b_vals, tool_rec, marker='s', linestyle=':', linewidth=2.0, color=PALETTE['base9'], label='Tool-Call Recall (≥82.86% Gate)')
    ax3.plot(b_vals, w2c_acc, marker='o', linewidth=2.5, color=PALETTE['ours'], label='When2Call Accuracy')
    ax3.plot(b_vals, info_rec, marker='D', linestyle='-.', linewidth=1.8, color=PALETTE['diff26'], label='Clarification Recall (≥78.44% Gate)')
    ax3.plot(b_vals, cannot_rec, marker='^', linestyle='--', linewidth=2.0, color=PALETTE['clef27'], label='Cannot-Answer Recall')

    # Uncalibrated (0,0,0,0) baseline marker
    ax3.scatter([0.00], [77.08], s=110, marker='*', color='#0f172a', zorder=6, label='No-Offset Baseline (77.08%)')

    ax3.axvline(0.34, color='#059669', linestyle='--', alpha=0.65, linewidth=1.5)
    ax3.annotate('Selected b_cannot=0.34\n(81.13% · All Gates Pass)', xy=(0.34, 81.13), xytext=(0.06, 85.8),
                 arrowprops=dict(arrowstyle='->', color='#059669', lw=1.4),
                 fontsize=8.6, fontweight='bold', color='#065f46')

    ax3.grid(True, zorder=0)
    ax3.set_xlim(-0.02, 0.50)
    ax3.set_ylim(65, 90)
    ax3.tick_params(axis='both', labelsize=9.2)
    ax3.set_xlabel('Abstention Offset b_cannot (b_tool=b_info=0.18)', fontsize=9.5, fontweight='bold')
    ax3.set_ylabel('Accuracy / Class Recall (%)', fontsize=10.2, fontweight='bold')
    ax3.set_title('c  Univariate b_cannot Sweep on When2Call', fontsize=11.2, fontweight='bold', loc='left', pad=8)
    ax3.legend(loc='lower right', frameon=True, fontsize=7.9)

    return save_both(fig, 'fig4_ablation_and_sensitivity')


def build_fig5_g4_6000_latency():
    """Fig 5: Hardware Profiling on NVIDIA RTX PRO 6000 Blackwell (G4 6000, 96 GB GDDR7)."""
    fig = plt.figure(figsize=(15.8, 5.0), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.20, 0.98, 0.96], wspace=0.25, left=0.065, right=0.985, top=0.88, bottom=0.18)

    # Panel A: Per-Request Amortized Processing Time (p50 & p95) and Token Throughput Across All 10 Benchmarks
    ax1 = fig.add_subplot(gs[0])
    bench_lat = [
        ('When2Call (794 tok)', 61.65, 124.53, 12862),
        ('PhishNChips (1,130 tok)', 89.37, 98.94, 12647),
        ('Amazon ESCI (442 tok)', 134.02, 284.86, 3349),
        ('BFCL (812 tok)', 222.49, 618.35, 3417),
        ('BANKING77 (1,864 tok)', 532.60, 535.41, 3496),
        ('ToolRet (9,263 tok)', 672.51, 1160.37, 13715),
        ('CLINC150 (3,391 tok)', 940.03, 942.19, 3607),
        ('Home Appl. (3,202 tok)', 969.76, 1028.15, 3304),
        ('BRIGHT (12,764 tok)', 1027.40, 2144.58, 12809),
        ('API-Bank (6,573 tok)', 1801.99, 1928.98, 3632),
    ]
    labels_b = [x[0] for x in bench_lat]
    p50_b = [x[1] for x in bench_lat]
    p95_b = [x[2] for x in bench_lat]

    y = np.arange(len(labels_b))
    h = 0.37
    ax1.barh(y + h/2, p50_b, height=h, color=PALETTE['ours'], label='Median Time (p50)', zorder=3)
    ax1.barh(y - h/2, p95_b, height=h, color=PALETTE['clef27'], alpha=0.80, label='Tail Time (p95)', zorder=3)

    for i, (m50, m95, tps) in enumerate(zip(p50_b, p95_b, [x[3] for x in bench_lat])):
        ax1.text(max(m50, m95) + 40, y[i], f'{m50:.0f} ms · {tps/1000:.1f}k tok/s',
                 va='center', ha='left', fontsize=8.4, fontweight='bold', color='#1e293b')

    ax1.grid(axis='x', zorder=0)
    ax1.set_xlim(0, 3450)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels_b, fontsize=8.8, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xlabel('Amortized Per-Request Time on G4 6000 (ms, B=16)', fontsize=9.6, fontweight='bold')
    ax1.set_title('a  10-Benchmark Processing Time & Throughput', fontsize=11.2, fontweight='bold', loc='left', pad=8)
    ax1.legend(loc='upper right', frameon=True, fontsize=8.5)

    # Panel B: Synthetic Scaling (Fixed L=1,024 Context) vs Estimated Autoregressive Decode (18 tok/question)
    ax2 = fig.add_subplot(gs[1])
    q_counts = np.array([1, 2, 4, 9, 16, 32])
    lat_fastbrain = np.array([62.0, 63.5, 66.2, 73.1, 82.5, 104.0])
    lat_ar9b = 60.0 + q_counts * 18 * 14.5
    lat_ar27b = 135.0 + q_counts * 18 * 28.0

    ax2.plot(q_counts, lat_ar27b, marker='v', linestyle='--', linewidth=2.1, color=PALETTE['jev27'],
             label='Est. Autoregressive 27B (18 tok/Q)')
    ax2.plot(q_counts, lat_ar9b, marker='s', linestyle='-.', linewidth=2.1, color=PALETTE['base9'],
             label='Est. Autoregressive 9B (18 tok/Q)')
    ax2.plot(q_counts, lat_fastbrain, marker='o', linewidth=2.6, color=PALETTE['ours'],
             label='FastBrain-9B (L=1,024 · 0 Output Tok)')

    ax2.annotate('32.9× vs 9B AR\n(K=9 Questions)', xy=(9, 73.1), xytext=(5.5, 600),
                 arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5),
                 fontsize=8.5, fontweight='bold', color='#065f46')
    ax2.annotate('80.9× vs 9B AR\n(K=32 Questions)', xy=(32, 104.0), xytext=(15.5, 1450),
                 arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5),
                 fontsize=8.5, fontweight='bold', color='#065f46')

    ax2.grid(True, zorder=0)
    ax2.set_yscale('log')
    ax2.set_ylim(45, 28000)
    ax2.set_xticks(q_counts)
    ax2.set_xticklabels([str(k) for k in q_counts], fontsize=9.2, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=9.2)
    ax2.set_xlabel('Schema Questions K per Request (Fixed L=1,024)', fontsize=9.6, fontweight='bold')
    ax2.set_ylabel('Latency (ms, Log Scale)', fontsize=10.0, fontweight='bold')
    ax2.set_title('b  Synthetic Multi-Question Scaling (L=1,024)', fontsize=11.2, fontweight='bold', loc='left', pad=8)
    ax2.legend(loc='upper left', frameon=True, fontsize=8.2)

    # Panel C: Observed Peak GPU Memory vs Batch Maximum Context Length on G4 RTX PRO 6000 (96 GB)
    ax3 = fig.add_subplot(gs[2])
    ctx_k = np.array([0.25, 0.5, 1.0, 2.0, 3.4, 6.5, 10.2, 16.0, 25.3, 31.5])
    vram_gb = np.array([20.27, 20.93, 22.12, 24.28, 24.71, 24.71, 24.71, 24.71, 24.71, 24.71])
    full_attn_vram = np.array([20.3, 21.2, 23.0, 26.8, 32.5, 45.2, 61.0, 86.4, 128.0, 156.0])

    ax3.plot(ctx_k, full_attn_vram, marker='s', linestyle='--', linewidth=1.9, color=PALETTE['red'],
             label='Unfused 32-Layer Full-Attn 9B (Est.)')
    ax3.plot(ctx_k, vram_gb, marker='o', linewidth=2.6, color=PALETTE['ours'],
             label='FastBrain-9B (Observed on G4 6000)')
    ax3.axhline(96.0, color='#334155', linestyle=':', linewidth=1.6, label='G4 6000 Capacity (96 GB)')

    ax3.fill_between(ctx_k, 0, vram_gb, color=PALETTE['ours'], alpha=0.12)
    ax3.annotate('Max Observed: 24.71 GB\n(25.7% of 96 GB G4 6000)', xy=(16.0, 24.71), xytext=(7.5, 49.0),
                 arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5),
                 fontsize=8.8, fontweight='bold', color='#065f46')

    ax3.grid(True, zorder=0)
    ax3.set_xlim(0, 33)
    ax3.set_ylim(15, 118)
    ax3.tick_params(axis='both', labelsize=9.2)
    ax3.set_xlabel('Batch Max Context Length (Thousands of Tokens)', fontsize=9.6, fontweight='bold')
    ax3.set_ylabel('Peak GPU Memory Allocated (GB)', fontsize=10.0, fontweight='bold')
    ax3.set_title('c  Observed Peak GPU Memory Scaling', fontsize=11.2, fontweight='bold', loc='left', pad=8)
    ax3.legend(loc='upper left', frameon=True, fontsize=8.1)

    return save_both(fig, 'fig5_g4_6000_latency_throughput')


def build_fig6_error_recovery():
    """Fig 6: Paired Sample-Level Error Recovery vs. New Errors Across All 8 Classification Benchmarks."""
    fig, ax = plt.subplots(figsize=(13.6, 4.5), dpi=300)

    datasets = [
        'CLINC150+OOS\n(n = 5,500)',
        'When2Call\n(n = 3,652)',
        'BANKING77\n(n = 3,080)',
        'PhishNChips\n(n = 2,000)',
        'API-Bank\n(n = 508)',
        'Amazon ESCI\n(n = 5,000)',
        'BFCL\n(n = 1,694)',
        'Home Appl.\n(n = 88)',
    ]
    recovered = np.array([1932, 618, 114, 90, 3, 6, 0, 0])
    new_errors = np.array([16, 43, 17, 15, 0, 8, 0, 0])
    net_gain = recovered - new_errors

    x = np.arange(len(datasets))
    w = 0.27
    ax.bar(x - w, recovered, width=w, color=PALETTE['ours'], label='Recovered Cases (Base Wrong → FastBrain-9B Correct)', zorder=3)
    ax.bar(x, new_errors, width=w, color=PALETTE['red'], alpha=0.85, label='New Case Errors (Base Correct → FastBrain-9B Wrong)', zorder=3)
    ax.bar(x + w, np.maximum(net_gain, 0), width=w, color=PALETTE['clef27'], label='Net Change (Δ Correct Cases)', zorder=4)

    for i in range(len(datasets)):
        y_rec = recovered[i] * 1.12 if recovered[i] >= 10 else recovered[i] + 0.45
        y_err = new_errors[i] * 1.12 if new_errors[i] >= 10 else new_errors[i] + 0.45
        y_net = max(net_gain[i], 0) * 1.12 if net_gain[i] >= 10 else max(net_gain[i], 0) + 0.45
        ax.text(x[i] - w, y_rec, f'{recovered[i]}', ha='center', va='bottom',
                fontsize=8.8, fontweight='bold', color='#065f46')
        ax.text(x[i], y_err, f'{new_errors[i]}', ha='center', va='bottom',
                fontsize=8.6, fontweight='bold', color='#b91c1c')
        net_str = f'+{net_gain[i]}' if net_gain[i] >= 0 else f'{net_gain[i]}'
        ax.text(x[i] + w, y_net, net_str, ha='center', va='bottom',
                fontsize=8.8, fontweight='bold', color='#1d4ed8')

    ax.grid(axis='y', zorder=0)
    ax.set_yscale('symlog', linthresh=20)
    ax.set_ylim(0, 3800)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9.2, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9.2)
    ax.set_ylabel('Number of Evaluation Requests (SymLog Scale)', fontsize=10.2, fontweight='bold')
    ax.set_title('Paired Request-Level Error Recovery vs. New Errors Across All 8 Classification Benchmarks (2,763 Recovered vs. 99 New Errors, Net +2,664)',
                 fontsize=10.8, fontweight='bold', loc='left', pad=8)
    ax.legend(loc='upper right', frameon=True, fontsize=9.0)

    return save_both(fig, 'fig6_per_benchmark_error_recovery')


def main():
    f1 = build_fig1_architecture()
    f2 = build_fig2_pareto_and_peers()
    f3 = build_fig3_governance_diagnostics()
    f4 = build_fig4_ablations()
    f5 = build_fig5_g4_6000_latency()
    f6 = build_fig6_error_recovery()
    print('All 6 figures built:', f1[1], f2[1], f3[1], f4[1], f5[1], f6[1])


if __name__ == '__main__':
    main()
