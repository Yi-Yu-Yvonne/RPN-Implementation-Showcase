# RPN-Implementation-Showcase

This repository contains a complete implementation of a Region Proposal Network (RPN) pipeline and a website showcasing the implementation.

## Repository Structure

- `/rpn-implementation`: Contains the Python implementation of the RPN pipeline
  - `main.py`: Main implementation file with all RPN functions
  - `rpn_utils.py`: Utility module with the same implementations
  - `test_rpn_utils.py`: Unit tests for the RPN functions
  - `report.md`: Detailed documentation of the implementation
  - `data/`: Directory containing test images and annotation files
  - `result/`: Directory containing generated tensor files

- `/website`: Contains the static website files showcasing the RPN implementation
  - Built with Next.js and exported as static HTML/CSS/JS
  - Includes detailed explanations, code snippets, and visualizations for each component

## RPN Pipeline Components

1. **Ground Truth Tensor Generation**: Generates patch-level ground truth tensors from annotated images
2. **Tensor Decoding**: Converts tensors back into bounding box proposals
3. **Anchor-Based Encoding**: Implements anchor-based encoding for better object localization
4. **Anchor-Based Decoding**: Decodes anchor-based predictions back into bounding boxes
5. **Non-Maximum Suppression**: Removes redundant overlapping predictions

## Live Website

A live version of the website is available at: https://ltgdgepq.manus.space

## Getting Started

### Running the RPN Implementation

1. Navigate to the `/rpn-implementation` directory
2. Install dependencies: `pip install pillow numpy matplotlib`
3. Run the tests: `python -m unittest test_rpn_utils.py`
4. Run the main program with different job numbers:
   - Job 1 (Generate ground truth tensors): `python main.py --job_number 1 --image_folder data/images_jpg --annotation_file data/coco.json --save`
   - Job 2 (Decode tensors): `python main.py --job_number 2 --image_folder data/images_jpg --annotation_file data/coco.json --tensor_folder result --threshold 0.65`
   - Job 3 (Anchor-based encoding): `python main.py --job_number 3 --image_folder data/images_jpg --annotation_file data/coco.json --anchor_file data/anchors.json --save`

### Viewing the Website Locally

The website is pre-built and can be viewed by opening `/website/index.html` in a web browser.

## License

This project is provided for educational purposes.
