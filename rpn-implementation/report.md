# Region Proposal Network (RPN) Implementation Report

## Overview

This report describes the implementation of a Region Proposal Network (RPN) pipeline including tensor generation, decoding, anchor encoding, and non-maximum suppression. The implementation follows the requirements specified in the assignment and passes all provided unit tests.

## Implementation Details

### 1. Ground Truth Tensor Generation

The implementation resizes input images to 200x200 and divides them into a 40×40 grid of 5×5 patches. For each patch:
- If any ground truth box overlaps the patch, it marks `Existence[row, col] = [1, 0]` and randomly selects one overlapping box to write its coordinates into the `Location` tensor.
- Otherwise, it marks `Existence[row, col] = [0, 1]` to indicate a background patch.

Key functions:
- `resize_image_and_boxes`: Resizes images and scales bounding boxes proportionally
- `compute_patch_grid`: Creates a grid of patches over the resized image
- `boxes_overlap`: Checks if two bounding boxes overlap
- `generate_gt_tensors`: Generates ground truth existence and location tensors

### 2. Decoding Ground Truth Tensors

The implementation converts existence and location tensors back into bounding box proposals for visualization and evaluation. For each grid cell:
- If `Existence[row, col, 0] > threshold`, it decodes the location tensor at that cell
- It converts `[x_center, y_center, width, height]` back to original image scale

Key function:
- `decode_tensors`: Converts tensors back into bounding boxes with confidence scores

### 3. Anchor-Based Encoding

The implementation labels each patch location and anchor shape based on how well it matches the ground truth boxes. For each anchor shape and patch:
- It places an anchor box at the patch center
- Computes IoU between the anchor and all ground truth boxes
- If the current anchor has the best IoU and is greater than zero, it sets `Existence[row, col] = [1, 0]` and encodes the ground truth box as offsets `[dx, dy, dw, dh]` from the anchor box
- Otherwise, it sets `Existence[row, col] = [0, 1]`

Key functions:
- `compute_iou`: Calculates Intersection over Union between boxes
- `match_anchors_to_ground_truth`: Matches anchors to ground truth boxes

### 4. Anchor-Based Decoding

The implementation decodes anchor-based existence and location tensors back into bounding box proposals using the predicted offsets and anchor configurations. For each anchor and patch location:
- If `Existence[row, col, 0] > threshold`, it decodes the box by computing the anchor center and applying offsets to get `[x_center, y_center, width, height]`
- It scales the decoded box from resized image (200x200) to original image size

Key function:
- `decode_anchor_grid_predictions`: Decodes anchor-based predictions

### 5. Non-Maximum Suppression (NMS)

The implementation improves the quality of decoded bounding boxes by applying Non-Maximum Suppression to remove redundant overlapping predictions. The algorithm:
- Sorts boxes by confidence score
- Iteratively keeps the highest-confidence box and removes others with IoU > threshold
- Produces a non-overlapping set of high-confidence proposals

Key function:
- `non_max_suppression`: Applies NMS to suppress redundant overlapping predictions

## Testing and Verification

All implemented functions pass the provided unit tests, confirming the correctness of the implementation. The main program has been verified to work correctly for:
- Job 1: Generating ground truth tensors
- Job 2: Decoding ground truth tensors
- Job 3: Performing anchor-based encoding

## Conclusion

The implementation successfully completes the core components of a Region Proposal Network pipeline. The code is modular, well-documented, and passes all provided tests, demonstrating a solid understanding of the RPN architecture and its components.
