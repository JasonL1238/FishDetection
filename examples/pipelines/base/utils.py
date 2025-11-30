"""
Shared utility functions for pipeline implementations.
"""


def count_fish_per_column(centroids, width, num_columns=7):
    """Count fish in each column."""
    column_width = width / num_columns
    column_counts = [0] * num_columns
    
    for y, x in centroids:
        column_idx = min(int(x / column_width), num_columns - 1)
        column_counts[column_idx] += 1
    
    return column_counts


def binary_search_min_size_columns(hsv_masker, bg_subtracted_frame, width, num_columns=7, 
                                    target_per_column=4, min_val=1, max_val=30, max_iterations=15):
    """Binary search to find min_object_size that gives target_per_column fish in each column."""
    best_min_size = min_val
    best_score = float('inf')
    best_column_counts = None
    
    for iteration in range(max_iterations):
        test_size = (min_val + max_val) // 2
        
        original_min_size = hsv_masker.min_object_size
        hsv_masker.min_object_size = test_size
        
        _, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_frame)
        column_counts = count_fish_per_column(centroids, width, num_columns)
        
        hsv_masker.min_object_size = original_min_size
        
        # Calculate score: sum of absolute differences from target
        score = sum(abs(count - target_per_column) for count in column_counts)
        
        # Check if all columns have exactly target_per_column
        if all(count == target_per_column for count in column_counts):
            return test_size, column_counts, iteration + 1
        
        # Track best result
        if score < best_score:
            best_min_size = test_size
            best_score = score
            best_column_counts = column_counts.copy()
        
        # Adjust search range
        total_fish = sum(column_counts)
        target_total = num_columns * target_per_column
        
        if total_fish > target_total:
            min_val = test_size + 1
        else:
            max_val = test_size - 1
        
        if min_val > max_val:
            break
    
    return best_min_size, best_column_counts, iteration + 1

