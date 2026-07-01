"""Insert Figure SX (mode analysis) cells into the notebook."""
import json, ast

NB_PATH = 'notebooks/generate_figures_and_tables_revised.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["##### Figure SX \u2014 Koopman Mode Analysis (Variance, Frequency, Ablation)"]
}

code_source = [
    "import json as _json\n",
    "\n",
    "with open(results_base_path / 'mode_frequencies.json') as _f:\n",
    "    _mf = _json.load(_f)\n",
    "with open(results_base_path / 'mode_ablation.json') as _f:\n",
    "    _ma = _json.load(_f)\n",
    "\n",
    "_modes  = _mf['modes']\n",
    "_PAL    = ['#dda251', '#5a286b', '#646464', '#ac4484']\n",
    "_C_COL  = _PAL[0]   # complex modes\n",
    "_R_COL  = _PAL[2]   # real modes\n",
    "_HL_COL = _PAL[3]   # dominant mode highlight\n",
    "\n",
    "# ordered labels, variances, colours for all 12 latent dims\n",
    "_labels, _variances, _colors = [], [], []\n",
    "for _m in _modes:\n",
    "    if _m['type'] == 'complex':\n",
    "        _labels.append('CP-{} ({:.2f} Hz)'.format(_m['index'], abs(_m['f_hz'])))\n",
    "        _variances.append(_m['latent_variance'])\n",
    "        _colors.append(_HL_COL if _m['index'] == 3 else _C_COL)\n",
    "    else:\n",
    "        _labels.append('R-{}'.format(_m['index']))\n",
    "        _variances.append(_m['latent_variance'])\n",
    "        _colors.append(_R_COL)\n",
    "\n",
    "_x = np.arange(len(_labels))\n",
    "\n",
    "# ablation data\n",
    "_abl       = sorted(_ma['ablation_results'], key=lambda r: r['mode_index'])\n",
    "_abl_lbls  = ['CP-{} ({:.2f} Hz)'.format(r['mode_index'], abs(r['f_hz'])) for r in _abl]\n",
    "_abl_delta = [r['pct_rmse_increase'] for r in _abl]\n",
    "_abl_cols  = [_HL_COL if r['mode_index'] == 3 else _C_COL for r in _abl]\n",
    "_baseline  = _ma['baseline_pct_rmse']\n",
    "\n",
    "_fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(16, 5))\n",
    "\n",
    "# Panel A: latent variance per mode\n",
    "_ax1.bar(_x, _variances, color=_colors, edgecolor='white', linewidth=0.8, zorder=3)\n",
    "_ax1.set_yscale('log')\n",
    "_ax1.set_xticks(_x)\n",
    "_ax1.set_xticklabels(_labels, rotation=40, ha='right', fontsize=9)\n",
    "_ax1.set_ylabel('Latent variance (log scale)', fontsize=12)\n",
    "_ax1.set_title('Latent Variance per Koopman Mode', fontsize=12)\n",
    "_ax1.yaxis.grid(True, which='both', linestyle='--', alpha=0.4)\n",
    "_ax1.set_axisbelow(True)\n",
    "_dom_i = _labels.index('CP-3 (1.77 Hz)')\n",
    "_ax1.annotate(\n",
    "    'Dominant (var={:.3f})'.format(_variances[_dom_i]),\n",
    "    xy=(_dom_i, _variances[_dom_i]),\n",
    "    xytext=(_dom_i - 2.2, _variances[_dom_i] * 0.5),\n",
    "    arrowprops=dict(arrowstyle='->', color='black', lw=1.2),\n",
    "    fontsize=9)\n",
    "from matplotlib.patches import Patch as _Patch\n",
    "_ax1.legend(handles=[\n",
    "    _Patch(facecolor=_C_COL, label='Complex pair'),\n",
    "    _Patch(facecolor=_HL_COL, label='Dominant complex pair'),\n",
    "    _Patch(facecolor=_R_COL,  label='Real mode'),\n",
    "], fontsize=9, frameon=False)\n",
    "\n",
    "# Panel B: |frequency| for complex modes\n",
    "_cp  = [_m for _m in _modes if _m['type'] == 'complex']\n",
    "_cpx = np.arange(len(_cp))\n",
    "_ax2.bar(_cpx, [abs(_m['f_hz']) for _m in _cp],\n",
    "         color=[_HL_COL if _m['index'] == 3 else _C_COL for _m in _cp],\n",
    "         edgecolor='white', linewidth=0.8, zorder=3)\n",
    "_ax2.set_yscale('log')\n",
    "_ax2.axhspan(0.833, 2.5, alpha=0.12, color='green',\n",
    "             label='Heart rate range (50-150 bpm)')\n",
    "_ax2.set_xticks(_cpx)\n",
    "_ax2.set_xticklabels(['CP-{}'.format(_m['index']) for _m in _cp], fontsize=11)\n",
    "_ax2.set_ylabel('|Frequency| Hz (log scale)', fontsize=12)\n",
    "_ax2.set_title('Complex Mode Frequencies', fontsize=12)\n",
    "_ax2.yaxis.grid(True, which='both', linestyle='--', alpha=0.4)\n",
    "_ax2.set_axisbelow(True)\n",
    "_ax2.legend(fontsize=9, frameon=False)\n",
    "for _xi, _m in zip(_cpx, _cp):\n",
    "    _fabs = abs(_m['f_hz'])\n",
    "    _lbl = 'T={:.2f} s'.format(_m['period_s']) if _fabs > 0.1 else 'T={:.0f} s'.format(_m['period_s'])\n",
    "    _ax2.text(_xi, _fabs * 1.6, _lbl, ha='center', fontsize=8.5)\n",
    "\n",
    "# Panel C: ablation delta %RMSE\n",
    "_ax3.bar(np.arange(len(_abl)), _abl_delta, color=_abl_cols,\n",
    "         edgecolor='white', linewidth=0.8, zorder=3)\n",
    "_ax3.axhline(_baseline, color='grey', linestyle='--', linewidth=1.2,\n",
    "             label='Baseline %RMSE ({:.1f}%)'.format(_baseline))\n",
    "_ax3.set_xticks(np.arange(len(_abl)))\n",
    "_ax3.set_xticklabels(_abl_lbls, rotation=20, ha='right', fontsize=9)\n",
    "_ax3.set_ylabel('Delta %RMSE (percentage points)', fontsize=12)\n",
    "_ax3.set_title('Mode Ablation: Impact on %RMSE', fontsize=12)\n",
    "_ax3.yaxis.grid(True, linestyle='--', alpha=0.4)\n",
    "_ax3.set_axisbelow(True)\n",
    "_ax3.legend(fontsize=9, frameon=False)\n",
    "for _xi, _r in zip(np.arange(len(_abl)), _abl):\n",
    "    _rel = _r['pct_rmse_increase'] / _baseline * 100\n",
    "    _ax3.text(_xi, _r['pct_rmse_increase'] + 0.15,\n",
    "              '+{:.0f}%'.format(_rel), ha='center', fontsize=9)\n",
    "\n",
    "plt.tight_layout()\n",
    "_out = results_base_path / 'figures' / 'figureSX_mode_analysis.svg'\n",
    "plt.savefig(_out, format='svg', bbox_inches='tight')\n",
    "plt.savefig(str(_out).replace('.svg', '.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Saved ->', _out)\n",
    "for _r in _abl:\n",
    "    print('  Ablate CP-{}: +{:.2f} pp ({:.1f}% relative)'.format(\n",
    "        _r['mode_index'], _r['pct_rmse_increase'],\n",
    "        _r['pct_rmse_increase'] / _baseline * 100))\n",
]

# syntax check
src = ''.join(code_source)
try:
    ast.parse(src)
    print('Syntax OK')
except SyntaxError as e:
    print(f'SyntaxError line {e.lineno}: {e.msg}')
    raise SystemExit(1)

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": code_source
}

nb['cells'].insert(38, code_cell)
nb['cells'].insert(38, md_cell)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# round-trip check
with open(NB_PATH, encoding='utf-8') as f:
    nb2 = json.load(f)
try:
    ast.parse(''.join(nb2['cells'][39]['source']))
    print(f'Round-trip OK — total cells: {len(nb2["cells"])}')
except SyntaxError as e:
    print(f'Round-trip SyntaxError: {e}')
