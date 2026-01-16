#!/bin/bash

# Quick testing script for cross_image_predictor.py
# Tests all prompt types with appropriate output names

# Set base command
BASE_CMD="uv run python scripts/cross_image_predictor.py --img1 assets/images/cat.jpg --img2 assets/images/cats_2.jpg"

# Create output directory if it doesn't exist
mkdir -p tests

echo "Testing cross image prediction with different prompt types..."
echo "============================================================"

# Test with mask prompt
echo "Testing mask prompt..."
$BASE_CMD --mask assets/images/cat-mask.jpg --output1 tests/cat_mask_image1.jpg --output2 tests/cat_mask_image2.jpg
echo "✓ Mask prompt results saved to tests/cat_mask_image1.jpg and tests/cat_mask_image2.jpg"
echo ""

# Test with point prompt
echo "Testing point prompt..."
$BASE_CMD --point 300 600 --output1 tests/cat_point_image1.jpg --output2 tests/cat_point_image2.jpg
echo "✓ Point prompt results saved to tests/cat_point_image1.jpg and tests/cat_point_image2.jpg"
echo ""

# Test with box prompt
echo "Testing box prompt..."
$BASE_CMD --box 80 171 450 840 --output1 tests/cat_box_image1.jpg --output2 tests/cat_box_image2.jpg
echo "✓ Box prompt results saved to tests/cat_box_image1.jpg and tests/cat_box_image2.jpg"
echo ""

# Test with text prompt
echo "Testing text prompt..."
$BASE_CMD --text "cat" --output1 tests/cat_text_image1.jpg --output2 tests/cat_text_image2.jpg
echo "✓ Text prompt results saved to tests/cat_text_image1.jpg and tests/cat_text_image2.jpg"
echo ""

echo "All tests completed! Check the tests/ directory for results."
