import numpy as np
import win32gui, win32ui, win32con
from threading import Thread, Lock
import time

class ScreenCapture:
    """
    A class to capture screenshots from a specific window or the desktop, 
    manage screen regions, and handle multi-threaded screenshot capturing.

    Attributes:
        stopped (bool): Indicates whether the screenshot capturing thread is stopped.
        lock (threading.Lock): A lock object to synchronize access to the screenshot.
        screenshot (ndarray): The most recent captured screenshot.
        w (int): Width of the capture area.
        h (int): Height of the capture area.
        hwnd (int): Handle to the target window for capturing.
        cropped_x (int): Offset in the x-axis to account for window borders.
        cropped_y (int): Offset in the y-axis to account for the window title bar.
        offset_x (int): X-offset of the window on the screen.
        offset_y (int): Y-offset of the window on the screen.
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
    hwnd = None  # Handle to the target window or desktop
    cropped_x = 0  # Horizontal cropping to account for window borders
    cropped_y = 0  # Vertical cropping to account for the window title bar
    offset_x = 0  # Horizontal offset of the window relative to the screen
    offset_y = 0  # Vertical offset of the window relative to the screen

    def __init__(self, window_name=None, window_rect=None):
        """
        Initialize the ScreenCapture object.

        Args:
            window_name (str, optional): Name of the target window. If None, captures the entire desktop.
            window_rect (tuple, optional): A tuple representing the window's rectangle coordinates (x1, y1, x2, y2).
                                            If None, automatically retrieves the window's size.

        Raises:
            Exception: If the specified window is not found.
        """
        # Create a thread lock object for synchronization
        self.lock = Lock()

        border_pixels = 0
        titlebar_pixels = 0

        if window_name is None:
            # Capture the entire desktop if no window name is provided
            self.hwnd = win32gui.GetDesktopWindow()
            if window_rect is None:
                window_rect = win32gui.GetWindowRect(self.hwnd)
        else:
            # Capture a specific window by its name
            self.hwnd = self.get_window_by_name(window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))
            
            # Account for window borders and title bar
            border_pixels = 8
            titlebar_pixels = 30
            window_rect = win32gui.GetWindowRect(self.hwnd)

        # Calculate the width and height of the window, excluding borders and title bar
        self.w = window_rect[2] - window_rect[0] - (border_pixels * 2)
        self.h = window_rect[3] - window_rect[1] - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # Calculate the screen position offset for translating coordinates
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):
        """
        Capture a screenshot of the target window or desktop.

        Returns:
            ndarray: The captured screenshot as a NumPy array (RGB format).
        """
        # Get the window's device context (DC) for capturing
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        
        # Perform a bit block transfer (bitblt) to capture the window's contents
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # Convert raw bitmap data into a NumPy array
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # Free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # Drop the alpha channel (RGBA -> RGB) for OpenCV compatibility
        img = img[...,:3]

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
            int: The handle of the window if found, otherwise None.
        """
        return win32gui.FindWindow(None, window_name)

    @staticmethod
    def list_window_names():
        """
        List the names and handles of all currently visible windows.
        """
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

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