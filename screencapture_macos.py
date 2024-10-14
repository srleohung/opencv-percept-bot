import cv2
import numpy as np
import Quartz.CoreGraphics as CG
from threading import Thread, Lock
import time


class ScreenCapture:
    """
    A class to capture screenshots from a specific window or the desktop on macOS, 
    manage screen regions, and handle multi-threaded screenshot capturing.

    Attributes:
        stopped (bool): Indicates whether the screenshot capturing thread is stopped.
        lock (threading.Lock): A lock object to synchronize access to the screenshot.
        screenshot (ndarray): The most recent captured screenshot.
        w (int): Width of the capture area.
        h (int): Height of the capture area.
        hwnd (dict): Handle to the target window for capturing (None for desktop).
        offset_x (int): X-offset of the window or screen region relative to the screen.
        offset_y (int): Y-offset of the window or screen region relative to the screen.
    """

    # **************************************************
    # * Threading Properties
    # **************************************************
    stopped = True  # Flag to control the capturing thread
    lock = None  # Threading lock to handle concurrent access to screenshot data
    screenshot = None  # Store the latest screenshot captured by the thread

    # **************************************************
    # * Window and Screen Properties
    # **************************************************
    w = 0  # Width of the capture area
    h = 0  # Height of the capture area
    hwnd = None  # Handle to the target window (or None for desktop capture)
    offset_x = 0  # Horizontal offset of the window or screen relative to the display
    offset_y = 0  # Vertical offset of the window or screen relative to the display

    def __init__(self, window_name=None, window_rect=None):
        """
        Initialize the ScreenCapture object.

        Args:
            window_name (str, optional): Name of the target window. If None, captures the entire desktop.
            window_rect (tuple, optional): A tuple representing the window's rectangle coordinates (x1, y1, x2, y2).
                                            If None, the entire screen is captured.

        Raises:
            Exception: If the specified window is not found.
        """
        # Create a thread lock object for synchronization
        self.lock = Lock()

        if window_name is None:
            # Capture the entire screen if no window name is provided
            self.hwnd = None
            if window_rect is None:
                # Get the main display's width and height
                display = CG.CGMainDisplayID()
                self.w = CG.CGDisplayPixelsWide(display)
                self.h = CG.CGDisplayPixelsHigh(display)
                
                # No offset needed for the full screen capture
                self.offset_x = 0
                self.offset_y = 0
            else:
                # Set the custom window rectangle size
                self.w = window_rect[2] - window_rect[0]
                self.h = window_rect[3] - window_rect[1]

                # Set the cropped coordinates offset for translating positions
                self.offset_x = window_rect[0]
                self.offset_y = window_rect[1]
        else:
            # Capture a specific window by its name
            self.hwnd = self.get_window_by_name(window_name)
            if not self.hwnd:
                raise Exception(f'Window not found: {window_name}')

            # Get the window bounds and set dimensions
            window_bounds = self.hwnd['kCGWindowBounds']
            self.w = window_bounds['Width']
            self.h = window_bounds['Height']

            # Set the cropped coordinates offset for translating positions
            self.offset_x = window_bounds['X']
            self.offset_y = window_bounds['Y']

    def get_screenshot(self):
        """
        Capture a screenshot of the target window or desktop.

        Returns:
            ndarray: The captured screenshot as a NumPy array (RGB format).
        """
        # Define the rectangle area to capture
        capture_rect = CG.CGRectMake(self.offset_x, self.offset_y, self.w, self.h)

        if self.hwnd is None:
            # Capture the entire screen
            image = CG.CGWindowListCreateImage(capture_rect, CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID, CG.kCGWindowImageDefault)
        else:
            # Capture a specific window
            image = CG.CGWindowListCreateImage(capture_rect, CG.kCGWindowListOptionIncludingWindow, int(self.hwnd['kCGWindowNumber']), CG.kCGWindowImageDefault)

        # Get image properties
        width = CG.CGImageGetWidth(image)
        height = CG.CGImageGetHeight(image)
        image_data = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))

        # Convert raw image data into a NumPy array
        img = np.frombuffer(image_data, dtype=np.uint8)

        # Adjust shape for OpenCV (height, width, 4 channels - RGBA)
        img = img.reshape((height, width, 4))

        # Drop the alpha channel (RGBA -> RGB) for OpenCV compatibility
        img = img[..., :3]

        # Ensure the image is contiguous in memory for OpenCV processing
        img = np.ascontiguousarray(img)

        return img

    @staticmethod
    def get_window_by_name(window_name):
        """
        Get the window handle (hwnd) by the window's name.

        Args:
            window_name (str): The title of the window.

        Returns:
            dict: The window information dictionary if found, otherwise None.
        """
        window_list = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionOnScreenOnly, 0)
        for window in window_list:
            if 'kCGWindowName' in window and window_name in window['kCGWindowName']:
                return window
        return None

    @staticmethod
    def list_window_names():
        """
        List the names of all currently visible windows.
        """
        window_list = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionOnScreenOnly, 0)
        for window in window_list:
            if 'kCGWindowName' in window and window['kCGWindowName']:
                print(window['kCGWindowName'])

    def get_screen_position(self, pos):
        """
        Translate a pixel position from a screenshot to the corresponding screen position.

        Args:
            pos (tuple): A tuple representing the pixel position (x, y) in the screenshot.

        Returns:
            tuple: The translated position (x, y) on the actual screen.
        """
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)

    # **************************************************
    # * Threading Methods
    # **************************************************

    def start(self):
        """
        Start a separate thread to continuously capture screenshots in the background.
        """
        self.stopped = False
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        """
        Stop the screenshot capturing thread.
        """
        self.stopped = True

    def run(self):
        """
        The main loop of the screenshot capturing thread. Continuously captures screenshots
        until stopped and stores the latest screenshot.
        """
        while not self.stopped:
            screenshot = self.get_screenshot()
            # Use a lock to safely update the screenshot in a multi-threaded environment
            self.lock.acquire()
            self.screenshot = screenshot
            self.lock.release()
            time.sleep(0.050)  # Add a slight delay to control capture rate