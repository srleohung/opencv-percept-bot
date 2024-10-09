import cv2
import platform
import pyautogui
import time
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

# initialize the ScreenCapture class
screencap = ScreenCapture(window_name)
screencap.start()

# callback function for handling mouse events in the window
def mouseCallback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = screencap.get_screen_position((x, y))
    elif event == cv2.EVENT_LBUTTONDOWN:
        pyautogui.moveTo(x=mouse_x, y=mouse_y)

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
        cv2.putText(display_image, f"FPS: {fps:.2f} X: {mouse_x} Y: {mouse_y}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.imshow(debug_window_name, display_image)
        cv2.setMouseCallback(debug_window_name, mouseCallback)

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        screencap.stop()
        cv2.destroyAllWindows()
        break
