import platform
import time
import argparse
import cv2
from utils import extract_text_from_image, parse_rectangle_string, convert_string_to_boolean
import pytesseract
import pyautogui
import importlib
import sys

# Setup argument parser
parser = argparse.ArgumentParser(description="A bot that captures screenshots from a window or specific screen area and triggers actions based on the visual input.")
parser.add_argument("--window_name", help="Specify the name of the window to capture. Leave blank to capture the entire screen.")
parser.add_argument("--window_rect", type=str, help="Specify the rectangle to capture as 'x,y,width,height'.", default="")
parser.add_argument("--debug", type=str, help="Enable or disable debug mode. Use 'true' or 'false'.", default="true")
parser.add_argument("--bot", help="Specify the name of the bot to use. Leave blank to use the default bot.")
subparsers = parser.add_subparsers(dest="command", help="Available commands")
subparsers.add_parser("list_window_names", help="List the names of all currently active windows.")
args = parser.parse_args()

# Dynamically import the specified bot module
if args.bot is None:
    from bot import Bot
else:
    try:
        module = importlib.import_module(args.bot)
        Bot = getattr(module, 'Bot')
    except ModuleNotFoundError:
        print(f"Error: The module '{args.bot}' does not exist.")
        sys.exit(1)  # Exit if the module is not found
    except AttributeError:
        print(f"Error: The class 'Bot' does not exist in the module '{args.bot}'.")
        sys.exit(1)  # Exit if the class is not found

# Import platform-specific screen capturing modules
if platform.system() == "Windows":
    from screencapture_windows import ScreenCapture
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
elif platform.system() == "Darwin":
    from screencapture_macos import ScreenCapture

# **************************************************
# * Properties and Constants
# **************************************************
DEBUG = convert_string_to_boolean(args.debug)  # Enable debug mode for extra logging and visuals
window_name = args.window_name  # Window name to capture
window_rect = parse_rectangle_string(args.window_rect)  # Default capture area rectangle (left, top, width, height)
mouse_pos = (0, 0)  # Current mouse position
fps = 0  # Frames per second tracking
frame_count = 0  # Frame count for FPS calculation
start_time = time.time()  # Initial timestamp for FPS calculation
drawing = False  # Toggle for drawing a rectangle on mouse events
start_point = (0, 0)  # Rectangle starting point
end_point = (0, 0)  # Rectangle ending point

# Initialize the screen capture class and bot
screencap = ScreenCapture(window_name, window_rect)
bot = Bot()

def mouse_callback(event, x, y, flags, param):
    """
    Callback function to handle mouse events for selecting areas in the window.

    Args:
        event (int): Mouse event (click, move, release).
        x (int): X-coordinate of the mouse.
        y (int): Y-coordinate of the mouse.
        flags (int): Event flags.
        param (object): Additional parameters.
    """
    global mouse_pos, drawing, start_point, end_point
    mouse_pos = (x, y)
    
    # Handle mouse dragging to update the end point for the rectangle
    if event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Simulate right-click and move the mouse to the clicked screen position
        screen_x, screen_y = screencap.get_screen_position(mouse_pos)
        pyautogui.moveTo(x=screen_x, y=screen_y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Toggle drawing mode when left mouse button is clicked
        if drawing:
            drawing = False
            capture_content_in_range(start_point, end_point)
        else:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)

def capture_content_in_range(start, end):
    """
    Capture and extract text content from the selected screen region.

    Args:
        start (tuple): Starting (x, y) point of the rectangle.
        end (tuple): Ending (x, y) point of the rectangle.
    
    Returns:
        str: Extracted text content from the selected image region.
    """
    content = extract_text_from_image(screencap.screenshot, start, end)
    print(f'Captured screen from {start} to {end}')
    print('------------------------- Start -------------------------')
    print(content)
    print('------------------------- End -------------------------')
    return content

def draw_detected_rectangles(image, rectangles):
    """
    Draw rectangles on the image based on the detected regions.

    Args:
        image (numpy.ndarray): The image where the rectangles will be drawn.
        rectangles (list): List of detected rectangles [(x, y, width, height)].
    
    Returns:
        numpy.ndarray: The image with drawn rectangles.
    """
    line_color = (0, 255, 0)  # Green color for the rectangle (BGR format)
    line_type = cv2.LINE_4

    for (x, y, w, h) in rectangles:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(image, top_left, bottom_right, line_color, lineType=line_type)
    
    return image

# **************************************************
# * Main Application Loop
# **************************************************

if args.command == "list_window_names":
    screencap.list_window_names()
else:
    screencap.start()
    bot.start()

    while True:
        if screencap.screenshot is None:
            continue

        bot.update_screenshot(screencap.screenshot)

        if DEBUG:
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Display the screenshot with overlay information
            debug_window_name = f'{window_name} - Viewer'
            display_image = screencap.screenshot.copy()

            # Display mouse position and FPS on the image
            global_x, global_y = screencap.get_screen_position(mouse_pos)
            cv2.putText(display_image, f"FPS: {fps:.2f} x: {mouse_pos[0]}({global_x}) y: {mouse_pos[1]}({global_y})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

            # Draw the rectangle if in drawing mode
            if drawing:
                cv2.rectangle(display_image, start_point, end_point, (100, 100, 100), 1)
                start_x, start_y = screencap.get_screen_position(start_point)
                cv2.putText(display_image, f"{start_x}, {start_y}", start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                end_x, end_y = screencap.get_screen_position(end_point)
                cv2.putText(display_image, f"{end_x}, {end_y}", end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

            # Draw detected rectangles if available
            if bot.rectangles is not None:
                draw_detected_rectangles(display_image, bot.rectangles)

            # Show the window with the updated image
            cv2.imshow(debug_window_name, display_image)
            cv2.setMouseCallback(debug_window_name, mouse_callback)

        # Press 'q' to quit the application
        key = cv2.waitKey(50)
        if key == ord('q'):
            screencap.stop()
            bot.stop()
            cv2.destroyAllWindows()
            break