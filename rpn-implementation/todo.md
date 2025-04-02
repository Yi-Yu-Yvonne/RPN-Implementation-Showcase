# RPN Implementation Tasks

## Functions to Implement

- [x] 1. `resize_image_and_boxes`: Resize image to 200x200 and scale bounding boxes
- [x] 2. `compute_patch_grid`: Create a 40x40 grid of 5x5 patches
- [x] 3. `boxes_overlap`: Check if two bounding boxes overlap
- [x] 4. `generate_gt_tensors`: Generate ground truth tensors from bounding boxes
- [x] 5. `decode_tensors`: Convert tensors back to bounding boxes
- [x] 6. `compute_iou`: Calculate Intersection over Union between boxes
- [x] 7. `match_anchors_to_ground_truth`: Match anchors to ground truth boxes
- [x] 8. `decode_anchor_grid_predictions`: Decode anchor-based predictions
- [x] 9. `non_max_suppression`: Apply NMS to remove redundant boxes

## Testing

- [x] Run unit tests with test_rpn_utils.py
- [x] Verify functionality with main.py for each job

## Deliverables

- [x] Completed implementation of all functions
- [x] Passing all unit tests
- [x] Working main.py with all jobs
- [x] Documentation of implementation details
