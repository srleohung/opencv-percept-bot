import pytesseract
import cv2 
import numpy as np

def extract_text_from_image(image, top_left, bottom_right):
    """
    Extract text from a specific region of an image using OCR.

    Args:
        image (ndarray): The input image to process.
        top_left (tuple): Coordinates of the top-left corner (x, y) of the region.
        bottom_right (tuple): Coordinates of the bottom-right corner (x, y) of the region.

    Returns:
        str: The extracted text from the specified region.
    """
    # Ensure coordinates are in correct order (top-left, bottom-right)
    x1, y1 = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
    x2, y2 = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])
    
    # Crop the image to the region
    cropped_image = image[y1:y2, x1:x2]
    
    # Use OCR to extract text
    extracted_text = pytesseract.image_to_string(cropped_image)
    
    return extracted_text

def get_rectangle_from_points(top_left, bottom_right):
    """
    Calculate the rectangle coordinates from the top-left and bottom-right points.

    Args:
        top_left (tuple): Coordinates of the top-left corner (x, y).
        bottom_right (tuple): Coordinates of the bottom-right corner (x, y).

    Returns:
        tuple: A rectangle in the format (x, y, width, height), ensuring that
               coordinates are in the correct order.
    """
    # Ensure coordinates are in correct order (top-left, bottom-right)
    x1, y1 = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
    x2, y2 = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Return the rectangle as (x, y, width, height)
    return (x1, y1, width, height)

def detect_template_in_image(image, template=None, template_path=None, threshold=0.75, method=cv2.TM_CCOEFF_NORMED):
    """
    Detect the presence of a template image within a larger image using template matching.

    Args:
        image (ndarray): The larger image to search within.
        template (ndarray, optional): The template image to search for.
        template_path (str, optional): Path to the template image file if template is not provided.
        threshold (float, optional): The matching threshold (default 0.75).
        method (int, optional): OpenCV matching method (default cv2.TM_CCOEFF_NORMED).

    Returns:
        list: A list of detected rectangles [(x, y, w, h)] for the template.
    """
    if template is None:
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    # Run the template matching algorithm
    match_result = cv2.matchTemplate(image, template, method)

    # Get all the match positions exceeding the threshold
    locations = np.where(match_result >= threshold)
    locations = list(zip(*locations[::-1]))

    # Extract rectangles for detected matches
    rectangles = _get_rectangles_from_locations(locations, template.shape[1], template.shape[0])

    # Group overlapping rectangles
    grouped_rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

    return grouped_rectangles

def _get_rectangles_from_locations(locations, width, height):
    """
    Helper function to generate rectangles from template match locations.

    Args:
        locations (list): List of (x, y) coordinates where template matches were found.
        width (int): Width of the template.
        height (int): Height of the template.

    Returns:
        list: List of rectangles in the format [(x, y, w, h)].
    """
    rectangles = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), width, height]
        rectangles.append(rect)
        rectangles.append(rect)  # Adding twice for groupRectangles
    return rectangles