import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 0; sigma = 1; x = np.linspace(-4, 4, 1000); y = norm.pdf(x, mu, sigma);
mad = sigma * np.sqrt(2 / np.pi)
outlier_threshold = 3 * sigma
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, color='black', lw=2)
ax.grid(True, linestyle='-', alpha=0.3)
ax.set_facecolor('white')
ax.axvline(mu, color='black', linestyle='-', lw=1.5) 
ax.axvline(mu + mad, color='black', linestyle='--', lw=1); ax.axvline(mu - mad, color='black', linestyle='--', lw=1)
ax.fill_between(x, y, where=(x >= mu - mad) & (x <= mu + mad), color='gray', alpha=0.1)
ax.fill_between(x, y, where=(x >= outlier_threshold), color='red', alpha=0.3); ax.fill_between(x, y, where=(x <= -outlier_threshold), color='red', alpha=0.3)
ax.text(0, 0.15, 'MAD', fontsize=12, fontweight='bold', ha='center'); ax.text(3.5, 0.02, 'Outliers', color='red', fontsize=10, ha='center'); ax.text(-3.5, 0.02, 'Outliers', color='red', fontsize=10, ha='center')
ax.set_title('Distribuția Normală: MAD și Outlieri', loc='left', fontsize=14); ax.set_ylabel('Density'); ax.set_xlabel('x')
xticks = [-outlier_threshold, -mad, 0, mad, outlier_threshold]
xtick_labels = [r'$-3\sigma$', '-MAD', r'$\bar{x}$', '+MAD', r'$3\sigma$']
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels)

plt.tight_layout()
plt.show()