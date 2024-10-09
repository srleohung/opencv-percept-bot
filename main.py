import platform
import time

# opencv
import cv2
import pytesseract

# bot
import pyautogui

# Check if the platform is Windows or macOS (Darwin)
if platform.system() == "Windows":
    from screencapture_windows import ScreenCapture
elif platform.system() == "Darwin":
    from screencapture_macos import ScreenCapture

# **************************************************
# * Properties
# **************************************************
DEBUG = True
window_name = 'Google Chat'

# initialize mouse position and FPS variables
mouse_x, mouse_y = 0, 0
fps = 0
frame_count = 0
start_time = time.time()
drawing = False  # True if the mouse is pressed
start_point = (0, 0)  # starting point for rectangle
end_point = (0, 0)  # ending point for rectangle

# initialize the ScreenCapture class
screencap = ScreenCapture(window_name)
screencap.start()

# callback function for handling mouse events in the window
def mouseCallback(event, x, y, flags, param):
    global mouse_x, mouse_y
    global drawing, start_point, end_point
    # https://steam.oxxostudio.tw/category/python/ai/opencv-mouseevent.html
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = screencap.get_screen_position((x, y))
        if drawing:
            end_point = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        pyautogui.moveTo(x=mouse_x, y=mouse_y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        if drawing:
            drawing = False
            image_to_string(start_point, end_point)
        else:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)


def image_to_string(start, end):
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


while (True):

    if screencap.screenshot is None:
        continue

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
        cv2.putText(display_image, f"FPS: {fps:.2f} x: {mouse_x} y: {mouse_y}",
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

        cv2.imshow(debug_window_name, display_image)
        cv2.setMouseCallback(debug_window_name, mouseCallback)

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    key = cv2.waitKey(50)
    if key == ord('q'):
        screencap.stop()
        cv2.destroyAllWindows()
        break
