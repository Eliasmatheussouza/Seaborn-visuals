import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_advanced_distribution_plot(n_samples=1000, save_path=None):
    """
    Creates an advanced visualization combining multiple Seaborn features:
    - Kernel Density Estimation (KDE) plot
    - Box plot
    - Strip plot
    - Custom styling and color palette
    - Multiple distribution comparison
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate for each distribution
    save_path : str, optional
        Path to save the figure. If None, the figure will be displayed
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Set the style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Generate sample data
    np.random.seed(42)
    categories = ['A', 'B', 'C']
    distributions = {
        'Normal': np.random.normal(0, 1, n_samples),
        'Gamma': np.random.gamma(2, 2, n_samples),
        'Exponential': np.random.exponential(2, n_samples)
    }
    
    # Create DataFrame
    data = []
    for dist_name, values in distributions.items():
        for category in categories:
            # Add some variation based on category
            category_offset = {'A': 0, 'B': 1, 'C': 2}
            modified_values = values + category_offset[category]
            
            data.extend([{
                'Distribution': dist_name,
                'Category': category,
                'Value': val
            } for val in modified_values])
    
    df = pd.DataFrame(data)
    
    # Create figure with custom size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create violin plot with inner box plot
    sns.violinplot(data=df, x='Category', y='Value', hue='Distribution',
                  split=True, inner='box', palette='husl',
                  cut=0, bw=.2)
    
    # Add strip plot for individual points
    sns.stripplot(data=df, x='Category', y='Value', hue='Distribution',
                 size=3, alpha=0.1, palette='husl',
                 jitter=0.05, dodge=True)
    
    # Customize the plot
    plt.title('Multi-Distribution Comparison Across Categories', 
              fontsize=14, pad=20)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    unique_labels = dict(zip(labels[:3], handles[:3]))
    ax.legend(unique_labels.values(), unique_labels.keys(),
             title='Distribution Type',
             bbox_to_anchor=(1.15, 1),
             loc='upper right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# Example usage
if __name__ == "__main__":
    # Create and display the plot
    fig = create_advanced_distribution_plot()
    plt.show()
    
    # Optional: Save the plot
    # fig = create_advanced_distribution_plot(save_path='advanced_distribution.png')