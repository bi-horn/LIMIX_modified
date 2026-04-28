'''Newly added - not part of the original pipeline'''

import seaborn as sns
import matplotlib.pyplot as plt

def pairwise_distributions(samples_df, 
                                 column_labels=None,
                                 hist_color="blue",
                                 hist_fill="blue",
                                 label_size=10,
                                 scatter_color="green",
                                 kde_color="orange",
                                 kde_fill=False,
                                 scatter_alpha=0.5,
                                 figsize=(12, 12)):
    """
    Plot pairwise distributions of features in a DataFrame using histograms, 
    KDE plots, and scatter plots.

    Parameters:
    ----------
    samples_df : pandas.DataFrame
        The DataFrame containing the data to visualize.
    
    column_labels : list, optional
        A list of column labels. If None, the existing DataFrame column names are used.
    
    hist_color : str, default "blue"
        Color for the histogram.
    
    hist_fill : str or bool, default "blue"
        Fill color for the histogram (if set to True, will fill the histogram).
    
    label_size : int, default 10
        Font size for the axis labels.
    
    scatter_color : str, default "green"
        Color for scatter plot points.
    
    kde_color : str, default "orange"
        Color for the KDE plot.
    
    kde_fill : bool, default False
        Whether to fill the KDE plot.
    
    scatter_alpha : float, default 0.5
        Transparency level for scatter plot points.
    
    figsize : tuple, default (12, 12)
        Size of the figure (width, height) in inches.
    """
        
    num_vars = samples_df.shape[1]
    fig, axes = plt.subplots(nrows=num_vars, ncols=num_vars, figsize=figsize)

    # Check and set column labels
    if column_labels is not None:
        if len(column_labels) != num_vars:
            raise ValueError("Length of column_labels must match the number of columns in samples.")
    else:
        column_labels = samples_df.columns.tolist()  # Convert index to list

    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                # Diagonal: KDE plot of the feature
                sns.kdeplot(samples_df.iloc[:, i], ax=axes[i, j], color=hist_color, alpha=0.5, fill=hist_fill)
                # Diagonal: Histogram of the feature
                sns.histplot(samples_df.iloc[:, i], ax=axes[i, j], color=hist_color, kde=True, fill=hist_fill)
                axes[i, j].set_ylabel('Probability Density', fontsize=label_size)
            elif i < j:
                # Upper triangle: Scatter plot and KDE plot
                axes[i, j].scatter(samples_df.iloc[:, j], samples_df.iloc[:, i], color=scatter_color, alpha=scatter_alpha, edgecolor='k')
                sns.kdeplot(data=samples_df, x=samples_df.columns[j], y=samples_df.columns[i], ax=axes[i, j], color=kde_color, fill=kde_fill)
                # Label the y-axis for KDE
                axes[i, j].set_ylabel('Probability Density', fontsize=label_size)
            else:
                # Hide the lower triangle
                axes[i, j].set_visible(False)
            
            # Label the x-axis and y-axis
            if i == num_vars - 1:
                axes[i, j].set_xlabel(column_labels[j] if column_labels else f'Feature {j+1}', fontsize=label_size)
            else:
                axes[i, j].set_xlabel('')
            
            if j == 0:
                axes[i, j].set_ylabel('Probability Density', fontsize=label_size)
            else:
                axes[i, j].set_ylabel('')

    # Ensure that labels are correctly set
    for i in range(num_vars):
        axes[i, i].set_xlabel(column_labels[i] if column_labels else f'Feature {i+1}', fontsize=label_size)
        axes[i, i].set_ylabel('Probability Density', fontsize=label_size)

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_pairwise_distributions(samples_df)