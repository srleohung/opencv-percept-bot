import platform
import time

# argparse
import argparse

# initialize the argument parser
parser = argparse.ArgumentParser(description="A bot that takes screenshots from a window or specific screen area to trigger actions.")
parser.add_argument("--window_name", help="Specify the name of the window to capture. Leave blank to capture the entire screen.")
subparsers = parser.add_subparsers(dest="command", help="Available commands")
subparsers.add_parser("list_window_names", help="List the names of all currently active windows.")
args = parser.parse_args()

# opencv
import cv2
import pytesseract

# bot
import pyautogui
from bot import Bot

# Check if the platform is Windows or macOS (Darwin)
if platform.system() == "Windows":
    from screencapture_windows import ScreenCapture
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
elif platform.system() == "Darwin":
    from screencapture_macos import ScreenCapture

# **************************************************
# * Properties
# **************************************************
DEBUG = True
window_name = args.window_name

# initialize mouse position and FPS variables
mouse_pos = (0, 0)
fps = 0
frame_count = 0
start_time = time.time()
drawing = False  # True if the mouse is pressed
start_point = (0, 0)  # starting point for rectangle
end_point = (0, 0)  # ending point for rectangle

# initialize the ScreenCapture class
screencap = ScreenCapture(window_name)
bot = Bot()

def mouseCallback(event, x, y, flags, param):
    # callback function for handling mouse events in the window
    global mouse_pos, drawing, start_point, end_point
    # https://steam.oxxostudio.tw/category/python/ai/opencv-mouseevent.html
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)
        if drawing:
            end_point = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        x, y = screencap.get_screen_position(mouse_pos)
        pyautogui.moveTo(x=x, y=y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        if drawing:
            drawing = False
            print_content_from_image_range(start_point, end_point)
        else:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)

def print_content_from_image_range(start, end):
    p1_x, p1_y = start
    p2_x, p2_y = end
    if start[0] > end[0]:
        p1_x = end[0]
        p2_x = start[0]
    if start[1] > end[1]:
        p1_y = end[1]
        p2_y = start[1]
    cropped_screenshot = screencap.screenshot[p1_y:p2_y, p1_x:p2_x]
    content = pytesseract.image_to_string(cropped_screenshot)
    if DEBUG:
        print(f'captured screen from {start} to {end}')
        print('------------------------- start -------------------------')
        print(content)
        print('-------------------------  end  -------------------------')
    return content

def draw_rectangles(image, rectangles):
        # these colors are actually BGR
        line_color = (0, 255, 0)
        line_type = cv2.LINE_4

        for (x, y, w, h) in rectangles:
            # determine the box positions
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # draw the box
            cv2.rectangle(image, top_left, bottom_right, line_color, lineType=line_type)

        return image


# Check which command is provided and take the appropriate action
if args.command == "list_window_names":
    screencap.list_window_names()
else:
    screencap.start()
    bot.start()
    while (True):

        if screencap.screenshot is None:
            continue

        bot.update_screenshot(screencap.screenshot)

        if DEBUG:
            # calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # display the images
            debug_window_name = f'{window_name} - Viewer'
            display_image = screencap.screenshot.copy()
            global_x, global_y = screencap.get_screen_position(mouse_pos)
            cv2.putText(display_image, f"FPS: {fps:.2f} x: {mouse_pos[0]}({global_x}) y: {mouse_pos[1]}({global_y})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            if drawing:
                # ensure start_point and end_point are integers
                cv2.rectangle(display_image, start_point,
                            end_point, (100, 100, 100), 1)
                start_x, start_y = screencap.get_screen_position(start_point)
                cv2.putText(display_image, f"{start_x}, {start_y}",
                            start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                end_x, end_y = screencap.get_screen_position(end_point)
                cv2.putText(display_image, f"{end_x}, {end_y}",
                            end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            if bot.rectangles is not None:
                draw_rectangles(display_image, bot.rectangles)
            cv2.imshow(debug_window_name, display_image)
            cv2.setMouseCallback(debug_window_name, mouseCallback)

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        key = cv2.waitKey(50)
        if key == ord('q'):
            screencap.stop()
            bot.stop()
            cv2.destroyAllWindows()
            break

