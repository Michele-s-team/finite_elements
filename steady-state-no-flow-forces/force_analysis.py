import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
# Load the CSV file into a DataFrame
variables = pd.read_csv('output/variables.csv')

# Extract unique values of k
k_values = variables['k'].unique()
print(k_values)
# Prepare the plot
plt.figure(figsize=(10, 6))

# Store slopes for each k
slopes = {}

# Define a tolerance for considering duplicates
tolerance = 1e-3

# Round the values of 'k' and ' w' to the specified tolerance
variables['k_rounded'] = variables['k'].round(-int(np.log10(tolerance)))
variables['w_rounded'] = variables[' w'].round(-int(np.log10(tolerance)))


# Extract unique pairs of (k, w) within the tolerance
kw_pairs = variables[['k_rounded', 'w_rounded']].drop_duplicates()

print(kw_pairs)
# Group data by unique (k_rounded, w_rounded) pairs
grouped_data = variables.groupby(['k_rounded', 'w_rounded'])

for (k, w), group in grouped_data:
    h = group[' h'].values.reshape(-1, 1)
    Fz = group[' Fz'].values

    # Perform linear regression
    model = LinearRegression()
    model.fit(h[0:2], Fz[0:2])
    slope = model.coef_[0]
    if w in slopes:
        slopes[w].append((k, slope))
    else:
        slopes[w] = [(k, slope)]

    # Plot F_z vs z_r
    plt.plot(h, Fz, 'o', label=f'k={k}, w={w}')
    #plt.plot(h, model.predict(h), '-', label=f'Fit k={k}, w={w}')


# Finalize the plot
plt.xlabel('h')
plt.ylabel('Fz')
plt.title('Dependence of F_z on h for different k')
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.legend(loc='upper left')
plt.show()

# Plot the slope of F(z_r) as a function of k
for w in slopes:
    slopes[w] = sorted(slopes[w], key=lambda x: x[0])
    w_values, slope_values = zip(*slopes[w])
    plt.plot(w_values, slope_values, 'o-', label=f'k={w}')

plt.xlabel('k')
plt.ylabel('Slope of F(z_r)')
plt.title('Variation of Slope with k')
plt.grid()
plt.legend()
plt.show()