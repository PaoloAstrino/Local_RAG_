
# Advanced multi-panel visualization for NQ-Open hybrid retrieval (aligned with hybrid_weight_visualization.py)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Hybrid configs and their metrics (from your CSVs)
configs = ['0.1/0.9', '0.2/0.8', '0.3/0.7']
sparse_weights = [0.1, 0.2, 0.3]
dense_weights = [0.9, 0.8, 0.7]
recall_at_5 = [0.921, 0.920, 0.933]
precision_at_5 = [0.698, 0.71, 0.701]
mrr = [0.810, 0.813, 0.812]
answer_coverage = [0.98, 0.98, 0.987]
recall_at_20 = [0.97, 0.98, 0.98]
precision_at_20 = [0.652, 0.659, 0.641]
recall_at_50 = [0.96, 0.98, 0.98]
precision_at_50 = [0.627, 0.627, 0.601]
multi_answer_rate = [0.969, 0.971, 0.976]

x = np.arange(len(configs))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Natural Questions Hybrid Weight Optimization Results (NQ-Open, 300q, 10k passages)', fontsize=16, fontweight='bold')

# Graph 1: MRR Performance (discrete x-axis)
ax1.plot(np.array(sparse_weights)*100, mrr, 'o-', linewidth=3, markersize=8, color='#FF6B35', markerfacecolor='white', markeredgewidth=2, markeredgecolor='#FF6B35')
ax1.set_xlabel('Sparse Weight (%)', fontweight='bold')
ax1.set_ylabel('MRR', fontweight='bold')
ax1.set_title('Mean Reciprocal Rank vs Sparse Weight', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(min(mrr)-0.005, max(mrr)+0.005)
ax1.set_xticks([10, 20, 30])
ax1.set_xticklabels(['10', '20', '30'])
# Value annotations
for i, v in enumerate(mrr):
    ax1.annotate(f'{v:.3f}', (sparse_weights[i]*100, v), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

# Graph 2: Recall Performance Comparison (discrete x-axis)
ax2.plot(np.array(sparse_weights)*100, recall_at_5, 'o-', label='Recall@5', linewidth=3, markersize=8)
ax2.plot(np.array(sparse_weights)*100, recall_at_20, 's-', label='Recall@20', linewidth=3, markersize=8)
ax2.plot(np.array(sparse_weights)*100, recall_at_50, '^-', label='Recall@50', linewidth=3, markersize=8)
ax2.set_xlabel('Sparse Weight (%)', fontweight='bold')
ax2.set_ylabel('Recall Score', fontweight='bold')
ax2.set_title('Recall Performance at Different K Values', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(min(recall_at_5 + recall_at_20 + recall_at_50)-0.01, max(recall_at_5 + recall_at_20 + recall_at_50)+0.01)
ax2.set_xticks([10, 20, 30])
ax2.set_xticklabels(['10', '20', '30'])


# Graph 3: Answer Coverage & Multi-Answer Rate (bar, discrete x-axis, narrower bars)
bar_width = 3
bar_x = np.array(sparse_weights)*100
bars1 = ax3.bar(bar_x - bar_width/2, answer_coverage, width=bar_width, color='#2E8B57', alpha=0.7, edgecolor='black', label='Answer Coverage')
bars2 = ax3.bar(bar_x + bar_width/2, multi_answer_rate, width=bar_width, color='#4169E1', alpha=0.7, edgecolor='black', label='Multi-Answer Rate')
ax3.set_xlabel('Sparse Weight (%)', fontweight='bold')
ax3.set_ylabel('Rate', fontweight='bold')
ax3.set_title('Answer Coverage & Multi-Answer Rate', fontweight='bold')
ax3.set_ylim(min(min(answer_coverage), min(multi_answer_rate))-0.005, max(max(answer_coverage), max(multi_answer_rate))+0.005)
ax3.set_xticks([10, 20, 30])
ax3.set_xticklabels(['10', '20', '30'])
ax3.legend()
ax3.grid(True, alpha=0.3)
# Value annotations
for bar, v in zip(bars1, answer_coverage):
    height = bar.get_height()
    ax3.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, v in zip(bars2, multi_answer_rate):
    height = bar.get_height()
    ax3.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
