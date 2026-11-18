import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

D = ROOT + 'data/'
s = pd.read_csv(D + 'enron_learning_curve_summary.csv')
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COLORS = {'Naive Bayes': '#2a78d6', 'SVM': '#eb6834'}          # same colors as the SMS chart

fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for name, g in s.groupby('model'):
    ax.plot(g.n_group, g.f1, color=COLORS[name], lw=2, marker='o', ms=6, mfc=COLORS[name],
            mec=SURFACE, mew=1.5, label=name, zorder=3)
    last = g.iloc[-1]
    ax.annotate(f'{last.f1:.3f}', (last.n_group, last.f1), xytext=(8, 0), textcoords='offset points',
                va='center', color=INK, fontsize=9)
ax.set_xscale('log')
ticks = sorted(s.n_group.unique())
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f'{int(t):,}' for t in ticks]))
ax.minorticks_off()
ax.set_ylim(0.90, 1.0); ax.set_xlim(400, 27000)
ax.set_xlabel('Number of training emails', color=INK2, fontsize=9)
ax.set_ylabel('Spam F1 score', color=INK2, fontsize=9)
ax.set_title('Email spam: F1 score by training size', color=INK, fontsize=11, loc='left', pad=12)
ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8.5, length=0)
ax.legend(loc='lower right', frameon=False, fontsize=9, labelcolor=INK)
fig.text(0.01, 0.01, 'Own representation. Enron-Spam emails, 5-fold cross-validation on the training set, mean of 5 folds. Y-axis starts at 0.90.',
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
os.makedirs(ROOT + 'figures', exist_ok=True)
fig.savefig(ROOT + 'figures/enron_learning_curve.png', facecolor=SURFACE)
