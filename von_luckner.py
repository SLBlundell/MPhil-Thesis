import matplotlib.pyplot as plt
import numpy as np

# 1. Extracted Data (Focusing on the active "doom loop" region)
haircuts = np.array([41.2, 45.5, 56.5])
gdp_declines = np.array([4.0, 12.0, 36.0])
labels = ['2nd Quartile', '3rd Quartile', '4th Quartile']

# 2. Curve Fitting (Quadratic fit for smooth convexity)
poly_coeffs = np.polyfit(haircuts, gdp_declines, 2)
x_smooth = np.linspace(39, 58, 200)
y_smooth = np.polyval(poly_coeffs, x_smooth)

# 3. Create the Linear Reference Line (The Chord - Bounded)
# Strictly connecting the 2nd and 4th Quartile observations
chord_x = [haircuts[0], haircuts[-1]]
chord_y = [gdp_declines[0], gdp_declines[-1]]

# Slope and intercept for the shaded fill region calculations
linear_slope = (gdp_declines[-1] - gdp_declines[0]) / (haircuts[-1] - haircuts[0])
linear_intercept = gdp_declines[0] - linear_slope * haircuts[0]

# 4. Plot Styling (Academic/LaTeX format)
plt.rcParams.update({
    "font.family": "serif",        
    "axes.spines.top": False,      
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True              
})

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# 5. Plot the Lines and Data
# The Linear Reference Line (Dashed, strictly bounded chord)
ax.plot(chord_x, chord_y, color='gray', linestyle='--', linewidth=2, zorder=1,
        label='Linear Reference (Chord)')

# The Smooth Convex Curve
ax.plot(x_smooth, y_smooth, color='#2c3e50', linewidth=2.5, zorder=2, 
        label='Convex Trend (Accelerating Penalty)')

# Shade the area between the chord and convex curve (Strictly bounded)
x_fill = np.linspace(haircuts[0], haircuts[-1], 100)
y_smooth_fill = np.polyval(poly_coeffs, x_fill)
y_linear_fill = linear_slope * x_fill + linear_intercept
ax.fill_between(x_fill, y_smooth_fill, y_linear_fill, color='#3498db', alpha=0.15, zorder=1, label='Convexity Gap')

# The actual data points
ax.scatter(haircuts, gdp_declines, color="#a3a3a3", s=80, edgecolor='white', 
           linewidth=1.5, zorder=3, label='Observed Averages')

# 6. Annotate the Data Points (Using your adjusted text coordinates)
for i, label in enumerate(labels):
    if i == 0:
        xytext = (15, -10)
        ha = 'left'
    elif i == 1:
        xytext = (15, -15)  
        ha = 'left'
    else:
        xytext = (-15, 10)
        ha = 'right'
        
    ax.annotate(f'{label}',
                (haircuts[i], gdp_declines[i]),
                textcoords="offset points",
                xytext=xytext,
                ha=ha,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

# 7. Configure Axes and Grid
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
ax.set_xlabel('Bulow-Rogoff Haircut (%)', labelpad=10)
ax.set_ylabel('Average GDP Decline (%)', labelpad=10)

ax.set_xlim(39, 60)
ax.set_ylim(0, 40)

# 8. Export to PDF
plt.tight_layout()
file_name = 'haircut_gdp_figure.pdf'
plt.savefig(file_name, format='pdf', bbox_inches='tight')
print(f"Plot successfully saved as '{file_name}'")

plt.show()