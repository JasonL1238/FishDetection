# Background Subtraction Refactoring

## Overview

Successfully extracted background subtraction functionality from the optical flow implementation into a separate, reusable module within the `src/processing/` folder.

## What Was Done

### 1. Created New Processing Module Structure
- **Location**: `src/processing/`
- **Files Created**:
  - `__init__.py` - Package initialization
  - `background_subtractor.py` - Main BackgroundSubtractor class

### 2. BackgroundSubtractor Class Features

The new `BackgroundSubtractor` class provides:

#### **Core Functionality**
- `create_background_model()` - Creates background model from video frames
- `apply_background_subtraction()` - Applies background subtraction to individual frames
- `process_video_frames()` - Processes multiple frames in batch
- `save_background_model()` / `load_background_model()` - Persistence functionality
- `reset()` - Reset the background subtractor state

#### **Configurable Parameters**
- `threshold` - Binary threshold for mask creation (default: 25)
- `blur_kernel_size` - Gaussian blur kernel size (default: (3, 3))
- `blur_sigma` - Gaussian blur standard deviation (default: 0)
- `default_frame_indices` - Frame indices for background model creation

#### **Default Background Frame Indices**
```python
[0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]
```

### 3. Updated Optical Flow Implementation

#### **Modified Files**
- `examples/optical_flow_video_processor.py`

#### **Changes Made**
- Added import for `BackgroundSubtractor`
- Replaced inline background subtraction methods with calls to `BackgroundSubtractor`
- Updated constructor to accept background subtraction parameters
- Simplified `create_background_model()` and `apply_background_subtraction()` methods

#### **Before vs After**
```python
# BEFORE: Inline implementation
def create_background_model(self, video_path: str, frame_indices: list = None):
    # 30+ lines of background model creation code
    # ...

def apply_background_subtraction(self, frame: np.ndarray, background: np.ndarray):
    # 15+ lines of background subtraction code
    # ...

# AFTER: Clean abstraction
def create_background_model(self, video_path: str, frame_indices: list = None):
    return self.background_subtractor.create_background_model(video_path, frame_indices)

def apply_background_subtraction(self, frame: np.ndarray, background: np.ndarray = None):
    return self.background_subtractor.apply_background_subtraction(frame, background)
```

### 4. Testing and Validation

#### **Test Files Created**
- `scripts/tests/test_background_subtractor.py` - Comprehensive unit tests
- `examples/demo_background_subtractor.py` - Usage demonstration

#### **Test Results**
✅ All tests passed successfully:
- Background model creation
- Save/load functionality
- Individual frame processing
- Batch frame processing
- Reset functionality
- Integration with optical flow pipeline

## Usage Examples

### Basic Usage
```python
from processing.background_subtractor import BackgroundSubtractor

# Initialize
bg_subtractor = BackgroundSubtractor(threshold=25)

# Create background model
bg_subtractor.create_background_model("video.mp4")

# Process single frame
bg_subtracted = bg_subtractor.apply_background_subtraction(frame)

# Process multiple frames
bg_frames = bg_subtractor.process_video_frames("video.mp4", num_frames=10)
```

### Advanced Usage
```python
# Custom parameters
bg_subtractor = BackgroundSubtractor(
    threshold=30,
    blur_kernel_size=(5, 5),
    blur_sigma=1.0,
    default_frame_indices=[0, 1000, 2000, 3000]
)

# Save/load background model
bg_subtractor.save_background_model("bg_model.npy")
bg_subtractor2 = BackgroundSubtractor()
bg_subtractor2.load_background_model("bg_model.npy")
```

## Benefits

### 1. **Reusability**
- Background subtraction can now be used by any tracking method
- Consistent implementation across the codebase
- Easy to maintain and update

### 2. **Modularity**
- Clear separation of concerns
- Background subtraction logic isolated from optical flow
- Easier to test and debug

### 3. **Flexibility**
- Configurable parameters
- Support for different video sources
- Persistence functionality for background models

### 4. **Maintainability**
- Single source of truth for background subtraction
- Easier to add new features or fix bugs
- Cleaner, more readable code

## File Structure

```
src/processing/
├── __init__.py
└── background_subtractor.py

examples/
├── optical_flow_video_processor.py (updated)
└── demo_background_subtractor.py (new)

scripts/tests/
└── test_background_subtractor.py (new)

docs/
└── BACKGROUND_SUBTRACTION_REFACTOR.md (this file)
```

## Integration

The refactored code maintains full backward compatibility with existing optical flow implementations while providing a clean, reusable abstraction for background subtraction that can be used by other tracking methods in the future.
