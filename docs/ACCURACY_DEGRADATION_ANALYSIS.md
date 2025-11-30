# Accuracy Degradation Analysis

## Problem Summary

The detection accuracy decreases significantly over time when processing video segments:

- **1st 5min**: 95.92% perfect frames, avg min_size=15.21, iterations=1.40
- **2nd 5min**: 92.93% perfect frames, avg min_size=14.27, iterations=1.39  
- **3rd 5min**: 57.62% perfect frames, avg min_size=19.09, iterations=3.44
- **4th 5min**: 41.90% perfect frames, avg min_size=22.13, iterations=4.21

## Root Cause

Each segment creates its own background model using **fixed frame indices** that span the entire video:
```python
[0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]
```

### The Problem

For later segments:
- **3rd segment (frames 12000-17999)**: Background model uses frames 0-10000 (before segment) + some after
- **4th segment (frames 18000-23999)**: Background model uses frames 0-16000 (before segment) + some after

This causes:
1. **Background drift**: Lighting/water conditions change over 20 minutes
2. **Mismatched background**: Earlier frames don't represent later segments
3. **More false positives**: Requires larger `min_size` (15→22) to filter noise
4. **More iterations**: Binary search struggles (1.4→4.2 iterations)

## Evidence

- Standard deviation increases: 0.19 → 0.72
- Column variance increases: Column 4 std goes from 0.09 → 0.48
- Average min_size increases: 15.21 → 22.13 (trying to filter more noise)
- Binary search iterations increase: 1.40 → 4.21 (harder to find optimal size)

## Solutions

### Solution 1: Use Single Background Model (Recommended)
Use the same background model from the beginning of the video for all segments. This worked well for the first segment (95.92% accuracy).

**Pros:**
- Simple to implement
- Consistent background across all segments
- First segment already shows this works well

**Cons:**
- Assumes background doesn't change significantly
- May not handle lighting changes well

### Solution 2: Segment-Specific Frame Indices
Use frame indices within or near each segment for background model creation.

**Example for 3rd segment (12000-17999):**
```python
frame_indices = [12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500]
```

**Pros:**
- Background model matches the segment being processed
- Handles background changes over time

**Cons:**
- More complex implementation
- Need to ensure enough frames are available

### Solution 3: Adaptive Background Model
Use a sliding window or adaptive background that updates over time.

**Pros:**
- Handles gradual background changes
- Most robust solution

**Cons:**
- Most complex to implement
- May require significant code changes

## Recommended Implementation

For immediate improvement, use **Solution 1**: Create one background model from the beginning of the video and reuse it for all segments. This should restore accuracy to ~95% for all segments.

