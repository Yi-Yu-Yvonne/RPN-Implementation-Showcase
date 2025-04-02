"""
RPN Utils Module - Implementation of Region Proposal Network utilities

This module contains implementations for the core components of a Region Proposal Network (RPN)
including tensor generation, decoding, anchor encoding, and non-maximum suppression.
"""

import numpy as np
from PIL import Image

# Constants from main.py
PATCH_SIZE = 5
TARGET_SIZE = 200
GRID_NUMBER = int(TARGET_SIZE / PATCH_SIZE)
THRESHOLD = 0.65  # Confidence threshold for 'Existence: Yes'

def resize_image_and_boxes(image_path, boxes):
    """
    Resizes the input image to TARGET_SIZE x TARGET_SIZE and scales bounding boxes
    already defined in the bottom-left origin (x=col, y=row from bottom).

    Args:
        image_path (str): Path to the image file.
        boxes (list): List of bounding boxes [x, y, w, h] in bottom-left coordinate system.

    Returns:
        image (PIL.Image): Resized image.
        resized_boxes (list): List of resized boxes [x, y, w, h].
    """
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    
    # Resize the image to TARGET_SIZE x TARGET_SIZE
    image = image.resize((TARGET_SIZE, TARGET_SIZE))
    
    # Scale the bounding boxes
    resized_boxes = []
    for box in boxes:
        x_center, y_center, width, height = box
        
        # Scale the coordinates and dimensions
        new_x_center = x_center * TARGET_SIZE / original_width
        new_y_center = y_center * TARGET_SIZE / original_height
        new_width = width * TARGET_SIZE / original_width
        new_height = height * TARGET_SIZE / original_height
        
        resized_boxes.append([new_x_center, new_y_center, new_width, new_height])
    
    return image, resized_boxes

def compute_patch_grid():
    """
    Creates a grid of patches over a TARGET_SIZE x TARGET_SIZE image using GRID_NUMBER and PATCH_SIZE,
    assuming a coordinate system where (0, 0) is at the bottom-left.

    Returns:
        grid (list): List of tuples (i, j, x_center, y_center, w, h) for each patch, where:
            - (i, j): grid column and row index (with (0,0) at bottom-left)
            - (x_center, y_center): center coordinates of the patch
            - (w, h): width and height of the patch
    """
    grid = []

    for i in range(GRID_NUMBER):        # i: column index (x-direction)
        for j in range(GRID_NUMBER):    # j: row index (y-direction, bottom-up)
            # Calculate the center coordinates of the patch
            x_center = i * PATCH_SIZE + PATCH_SIZE / 2
            y_center = j * PATCH_SIZE + PATCH_SIZE / 2
            
            # Calculate the top-right corner of the patch
            w = x_center + PATCH_SIZE / 2
            h = y_center + PATCH_SIZE / 2
            
            # Add the patch to the grid
            grid.append((i, j, x_center, y_center, w, h))

    return grid

def boxes_overlap(box1, box2):
    """
    Checks whether two boxes overlap (non-zero intersection area),
    assuming (0, 0) is at the bottom-left and boxes are in center format.

    Args:
        box1, box2 (list or tuple): Bounding boxes in 
            [x_center, y_center, width, height] format, using bottom-left origin.

    Returns:
        bool: True if boxes overlap, False otherwise.
    """
    # Extract box coordinates
    x1_center, y1_center, w1, h1 = box1
    x2_center, y2_center, w2, h2 = box2
    
    # Calculate the half-widths and half-heights
    half_w1, half_h1 = w1 / 2, h1 / 2
    half_w2, half_h2 = w2 / 2, h2 / 2
    
    # Calculate the top-left and bottom-right corners of each box
    x1_min, y1_min = x1_center - half_w1, y1_center - half_h1
    x1_max, y1_max = x1_center + half_w1, y1_center + half_h1
    
    x2_min, y2_min = x2_center - half_w2, y2_center - half_h2
    x2_max, y2_max = x2_center + half_w2, y2_center + half_h2
    
    # Check for overlap
    x_overlap = x1_max > x2_min and x2_max > x1_min
    y_overlap = y1_max > y2_min and y2_max > y1_min
    
    return x_overlap and y_overlap

def generate_gt_tensors(boxes):
    """
    Generates ground-truth existence and location tensors for a resized image.

    Args:
        boxes (list): List of ground-truth bounding boxes in the format [x_center, y_center, width, height],
                      assuming a coordinate system with (0,0) at the bottom-left.

    Returns:
        existence (ndarray): Tensor of shape (GRID_NUMBER, GRID_NUMBER, 2), where
                             existence[j, i, 0] = 1 if an object exists in patch (i, j),
                             existence[j, i, 1] = 1 otherwise (i.e., background patch).
        location (ndarray): Tensor of shape (GRID_NUMBER, GRID_NUMBER, 4), where
                            location[j, i] = [x_center, y_center, width, height] of a GT box
                            assigned to patch (i, j), if any.
    """
    # Initialize the existence tensor with shape (grid_height, grid_width, 2)
    # It will hold one-hot encoded values: [1, 0] for object, [0, 1] for background
    existence = np.zeros((GRID_NUMBER, GRID_NUMBER, 2))
    
    # Initialize the location tensor to store box details for positive patches
    location = np.zeros((GRID_NUMBER, GRID_NUMBER, 4))
    
    # Generate the patch grid using bottom-left (0, 0) origin, returning:
    # (i, j, x_center, y_center, w, h) for each patch
    grid = compute_patch_grid()
    
    # Set all patches to background by default
    existence[:, :, 1] = 1
    
    # For each patch in the grid
    for i, j, patch_x_center, patch_y_center, patch_w, patch_h in grid:
        # Create a patch box in [x_center, y_center, width, height] format
        patch_box = [patch_x_center, patch_y_center, PATCH_SIZE, PATCH_SIZE]
        
        # Find all overlapping ground truth boxes
        overlapping_boxes = []
        for box in boxes:
            if boxes_overlap(patch_box, box):
                overlapping_boxes.append(box)
        
        # If there are overlapping boxes
        if overlapping_boxes:
            # Mark this patch as containing an object
            existence[j, i, 0] = 1
            existence[j, i, 1] = 0
            
            # Randomly pick one of the overlapping boxes
            import random
            selected_box = random.choice(overlapping_boxes)
            
            # Store the selected box's coordinates in the location tensor
            location[j, i] = selected_box
    
    return existence, location

def decode_tensors(existence, location, original_w, original_h, threshold=THRESHOLD):
    """
    Converts tensors back into bounding boxes by thresholding on the confidence map.

    Args:
        existence (ndarray): Existence tensor from model output.
        location (ndarray): Location tensor from model output.
        original_w (int): Width of the original image.
        original_h (int): Height of the original image.
        threshold (float): Confidence threshold to keep predicted boxes.

    Returns:
        boxes (list): List of decoded boxes in [cx, cy, w, h, confidence].
    """
    boxes = set([])
    
    # Iterate through each grid cell
    for i in range(GRID_NUMBER):
        for j in range(GRID_NUMBER):
            # Check if the existence score is above the threshold
            if existence[j, i, 0] > threshold:
                # Get the location data for this cell
                x_center, y_center, width, height = location[j, i]
                
                # Scale the coordinates back to the original image size
                scaled_x = x_center * original_w / TARGET_SIZE
                scaled_y = y_center * original_h / TARGET_SIZE
                scaled_w = width * original_w / TARGET_SIZE
                scaled_h = height * original_h / TARGET_SIZE
                
                # Add the box with its confidence score
                boxes.add((scaled_x, scaled_y, scaled_w, scaled_h, float(existence[j, i, 0])))
    
    return list(boxes)

def compute_iou(box1, box2):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes.

    Args:
        box1 (list or tuple): The first box in [cx, cy, w, h] format,
            where (cx, cy) is the center of the box, and (w, h) are width and height.
        box2 (list or tuple): The second box in [cx, cy, w, h] format.

    Returns:
        float: IoU score between box1 and box2, ranging from 0 (no overlap)
               to 1 (perfect overlap).
    """
    # Extract box coordinates
    x1_center, y1_center, w1, h1 = box1
    x2_center, y2_center, w2, h2 = box2
    
    # Calculate the half-widths and half-heights
    half_w1, half_h1 = w1 / 2, h1 / 2
    half_w2, half_h2 = w2 / 2, h2 / 2
    
    # Calculate the top-left and bottom-right corners of each box
    x1_min, y1_min = x1_center - half_w1, y1_center - half_h1
    x1_max, y1_max = x1_center + half_w1, y1_center + half_h1
    
    x2_min, y2_min = x2_center - half_w2, y2_center - half_h2
    x2_max, y2_max = x2_center + half_w2, y2_center + half_h2
    
    # Calculate the intersection area
    x_intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_intersection = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection_area = x_intersection * y_intersection
    
    # Calculate the union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou

def match_anchors_to_ground_truth(resized_boxes, grid, anchor_shape):
    """
    Matches anchors to ground truth boxes and generates supervision tensors.

    Args:
        resized_boxes (List[List[float]]): A list of ground truth boxes, where each box is in 
            [cx, cy, w, h] format and has been resized to match the current image dimensions.

        grid (List[Tuple[int, int, float, float, float, float]]): A list representing the patch grid.
            Each entry is a tuple (i, j, x_center, y_center, w, h), where (i, j) are grid indices and
            (x_center, y_center, w, h) define the center coordinates and size of each patch.

        anchor_shape (Tuple[float, float]): A tuple (ws, hs) representing the width and height scaling 
            factors (relative to the patch size) for the current anchor.

    Returns:
        existence (np.ndarray): A tensor of shape (GRID_NUMBER, GRID_NUMBER, 2), where each entry is [1, 0] 
            if the best-matching anchor overlaps with a ground truth box, otherwise remains [0, 0].
        location (np.ndarray): A tensor of shape (GRID_NUMBER, GRID_NUMBER, 4), containing the offsets 
            [dx, dy, dw, dh] between the matched anchor and the ground truth box for each patch.
    """
    # Initialize output tensors and best IoU tracker
    existence = np.zeros((GRID_NUMBER, GRID_NUMBER, 2))
    location = np.zeros((GRID_NUMBER, GRID_NUMBER, 4))
    best_iou_map = np.zeros((GRID_NUMBER, GRID_NUMBER))
    ws, hs = anchor_shape
    
    # Set all patches to background by default
    existence[:, :, 1] = 1
    
    # Iterate through each ground truth box and patch grid cell
    for gt in resized_boxes:
        for i, j, x_center, y_center, _, _ in grid:
            # Create anchor box with the specified shape at this patch center
            anchor_w = PATCH_SIZE * ws
            anchor_h = PATCH_SIZE * hs
            anchor_box = [x_center, y_center, anchor_w, anchor_h]
            
            # Compute IoU between anchor and ground truth box
            iou = compute_iou(anchor_box, gt)
            
            # If this anchor has better IoU than previous ones at this location
            if iou > best_iou_map[j, i]:
                best_iou_map[j, i] = iou
                
                # If there's any overlap (IoU > 0)
                if iou > 0:
                    # Mark as positive example
                    existence[j, i, 0] = 1
                    existence[j, i, 1] = 0
                    
                    # Calculate offsets (dx, dy, dw, dh) from anchor to ground truth
                    gt_x, gt_y, gt_w, gt_h = gt
                    
                    # Encode the offsets
                    dx = gt_x - x_center
                    dy = gt_y - y_center
                    dw = gt_w - anchor_w
                    dh = gt_h - anchor_h
                    
                    # Store the offsets in the location tensor
                    location[j, i] = [dx, dy, dw, dh]
    
    return existence, location

def decode_anchor_grid_predictions(existence, location, anchor_shape, original_size, threshold):
    """
    Decodes predicted anchor-based bounding box outputs from a grid-based format.

    This function takes predicted existence/confidence scores and location offsets 
    from a fixed grid of anchor boxes and converts them into bounding box coordinates 
    in the original image space. Only boxes with confidence scores above the specified 
    threshold are returned.

    Args:
        existence (np.ndarray): A (GRID_NUMBER, GRID_NUMBER, 2) array containing confidence 
                                scores for each grid cell.
        location (np.ndarray): A (GRID_NUMBER, GRID_NUMBER, 4) array containing predicted 
                               offsets (dx, dy, dw, dh) for each grid cell.
        anchor_shape (Tuple[float, float]): The normalized width and height (ws, hs) of the 
                                            anchor box for this prediction level.
        original_size (Tuple[int, int]): The original width and height of the input image.
        threshold (float): Confidence threshold for including a bounding box.

    Returns:
        Set[Tuple[float, float, float, float, float]]: A set of predicted bounding boxes 
        in the format (cx, cy, w, h, confidence), all scaled to the original image dimensions.
    """
    boxes = set()
    original_w, original_h = original_size
    ws, hs = anchor_shape
    
    # Iterate through each grid cell
    for i in range(GRID_NUMBER):
        for j in range(GRID_NUMBER):
            # Check if the existence score is above the threshold
            if existence[j, i, 0] > threshold:
                # Get the predicted offsets
                dx, dy, dw, dh = location[j, i]
                
                # Calculate the anchor center coordinates
                patch_cx = i * PATCH_SIZE + PATCH_SIZE / 2
                patch_cy = j * PATCH_SIZE + PATCH_SIZE / 2
                
                # Calculate the anchor dimensions
                anchor_w = PATCH_SIZE * ws
                anchor_h = PATCH_SIZE * hs
                
                # Apply offsets to get the predicted box
                pred_cx = patch_cx + dx
                pred_cy = patch_cy + dy
                pred_w = anchor_w + dw
                pred_h = anchor_h + dh
                
                # Scale to original image size
                scaled_cx = pred_cx * original_w / TARGET_SIZE
                scaled_cy = pred_cy * original_h / TARGET_SIZE
                scaled_w = pred_w * original_w / TARGET_SIZE
                scaled_h = pred_h * original_h / TARGET_SIZE
                
                # Add the box with its confidence score
                boxes.add((scaled_cx, scaled_cy, scaled_w, scaled_h, float(existence[j, i, 0])))
    
    return boxes

def non_max_suppression(boxes, threshold=THRESHOLD):
    """
    Applies Non-Maximum Suppression (NMS) to remove redundant overlapping bounding boxes.

    This function helps reduce duplicate detections by keeping only the most confident 
    bounding box among overlapping ones. It iteratively selects the box with the highest 
    confidence score and removes all other boxes that have high overlap (IoU) with it.

    Args:
        boxes (list): A list of bounding boxes, each represented as a 5-element list or tuple:
                      [cx, cy, w, h, confidence], where (cx, cy) is the center,
                      (w, h) are width and height, and `confidence` is the predicted score.
        threshold (float): The Intersection-over-Union (IoU) threshold used to suppress
                           overlapping boxes. If IoU between two boxes exceeds this value,
                           the box with lower confidence is removed.

    Returns:
        keep (list): A list of filtered bounding boxes after NMS, retaining only the
                     most confident non-overlapping predictions.
    """
    if not boxes:
        return []
    
    # Convert to list if it's not already
    boxes = list(boxes)
    
    # Sort boxes by confidence score (highest first)
    boxes.sort(key=lambda x: x[4], reverse=True)
    
    keep = []
    
    while boxes:
        # Take the box with highest confidence
        current_box = boxes.pop(0)
        keep.append(current_box)
        
        # Check remaining boxes
        i = 0
        while i < len(boxes):
            # If IoU with current_box is above threshold, remove the box
            if compute_iou(current_box[:4], boxes[i][:4]) > threshold:
                boxes.pop(i)
            else:
                i += 1
    
    return keep
