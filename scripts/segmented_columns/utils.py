"""
Utility functions for segmented columns pipeline.
"""


def count_fish_per_column(centroids, width, num_columns=7):
    """Count fish in each column."""
    column_width = width / num_columns
    column_counts = [0] * num_columns
    
    for y, x in centroids:
        column_idx = min(int(x / column_width), num_columns - 1)
        column_counts[column_idx] += 1
    
    return column_counts

