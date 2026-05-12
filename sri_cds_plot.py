import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# --- Academic style parameters ---
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         14,
    'axes.linewidth':    1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.size':  5,
    'ytick.major.size':  5,
    'figure.dpi':        300,
    'axes.grid':         False,
})

# 1. Load
df = pd.read_excel('data/sri_lanka.xlsx', sheet_name='Sheet1')
df.columns = ['Date', 'Spread']
df['Date']   = pd.to_datetime(df['Date'])
df['Spread'] = pd.to_numeric(df['Spread'], errors='coerce')
df = df.dropna().sort_values('Date')
MORATORIUM = pd.Timestamp('2022-04-12')
df = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= MORATORIUM)]

# Break line across gaps > 5 business days
df = df.set_index('Date')
df = df.reindex(pd.bdate_range(df.index.min(), df.index.max()))

# 2. Plot
fig, ax = plt.subplots(figsize=(7.5, 5.0))   # full-width A4 portrait

ax.plot(df.index, df['Spread'],
        color="#0046AF", linewidth=1.6, zorder=3)

# 3. Log scale
ax.set_yscale('log')
ax.set_ylim(bottom=50)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Y-axis: plain integer ticks at sensible bps levels
yticks = [100, 200, 500, 1000, 2000, 5000, 10000]
ax.set_yticks(yticks)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.yaxis.set_minor_locator(ticker.NullLocator())

# X-axis: yearly ticks
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(axis='x', rotation=0)

# 4. Moratorium marker
ax.axvline(MORATORIUM, color='#444444', linewidth=0.75, linestyle='--', zorder=2)
ax.text(MORATORIUM - pd.Timedelta(days=18), ax.get_ylim()[1],
        'Moratorium\n12 Apr 2022', fontsize=11, color='#444444',
        ha='right', va='top', fontstyle='italic')

# 5. Labels
ax.set_xlabel('', fontsize=14)
ax.set_ylabel('CDS spread (bps, log scale)', fontsize=14)
ax.set_title('Sri Lanka 5-Year Sovereign CDS Spread, 2019–2022',
             fontsize=14, fontweight='normal', pad=10)

# Subtle horizontal reference grid on y-axis only
ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#cccccc', zorder=1)

plt.tight_layout(pad=0.5)
plt.savefig('sri_lanka_cds.pdf', format='pdf', bbox_inches='tight')
plt.show()
