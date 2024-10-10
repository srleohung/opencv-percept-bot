import numpy as np
import Quartz.CoreGraphics as CG
from threading import Thread, Lock
import time


class ScreenCapture:

    # **************************************************
    # * Threading Properties
    # **************************************************
    stopped = True
    lock = None
    screenshot = None

    # **************************************************
    # * Properties
    # **************************************************
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name=None):
        # create a thread lock object
        self.lock = Lock()

       # find the handle for the window we want to capture.
        # if no window name is given, capture the entire screen
        if window_name is None:
            # Capture the entire screen if no window name is given
            self.hwnd = None
            display = CG.CGMainDisplayID()
            self.w = CG.CGDisplayPixelsWide(display)
            self.h = CG.CGDisplayPixelsHigh(display)
        else:
            self.hwnd = self.get_window_by_name(window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))

            # get the window size
            window_bounds = self.hwnd['kCGWindowBounds']
            self.w = window_bounds['Width']
            self.h = window_bounds['Height']

            # macOS typically doesn't require border/titlebar cropping
            self.cropped_x = 0
            self.cropped_y = 0

            # set the cropped coordinates offset for translating positions
            self.offset_x = window_bounds['X']
            self.offset_y = window_bounds['Y']
    
    def get_screenshot(self):
        
        # if capturing the entire screen
        if self.hwnd is None:
            screen_rect = CG.CGRectInfinite
            image = CG.CGWindowListCreateImage(screen_rect, CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID, CG.kCGWindowImageDefault)
        else:
            # capture a specific window
            window_bounds = self.hwnd['kCGWindowBounds']
            capture_rect = CG.CGRectMake(window_bounds['X'], window_bounds['Y'], window_bounds['Width'], window_bounds['Height'])
            image = CG.CGWindowListCreateImage(capture_rect, CG.kCGWindowListOptionIncludingWindow, int(self.hwnd['kCGWindowNumber']), CG.kCGWindowImageDefault)

        # get image width, height, and pixel data
        width = CG.CGImageGetWidth(image)
        height = CG.CGImageGetHeight(image)
        image_data = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))

        # convert to numpy array
        img = np.frombuffer(image_data, dtype=np.uint8)

        # adjust shape for OpenCV (height, width, 4 channels - RGBA)
        img = img.reshape((height, width, 4))

        # drop the alpha channel for OpenCV compatibility (RGBA to RGB)
        img = img[..., :3]

        # ensure the image is contiguous in memory
        img = np.ascontiguousarray(img)

        return img

    # get the window by name
    @staticmethod
    def get_window_by_name(window_name):
        window_list = CG.CGWindowListCopyWindowInfo(
            CG.kCGWindowListOptionOnScreenOnly, 0)
        for window in window_list:
            if 'kCGWindowName' in window and window_name in window['kCGWindowName']:
                return window
        return None
    
    # list the names of all current windows
    @staticmethod
    def list_window_names():
        window_list = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionOnScreenOnly, 0)
        for window in window_list:
            if 'kCGWindowName' in window and window['kCGWindowName']:
                print(window['kCGWindowName'])
    
    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    def get_screen_position(self, pos):
        return (pos[0] + self.hwnd['kCGWindowBounds']['X'], pos[1] + self.hwnd['kCGWindowBounds']['Y'])
    
    # **************************************************
    # * Threading Methods
    # **************************************************

    def start(self):
        self.stopped = False
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
            # get an updated image of the target window screen
            screenshot = self.get_screenshot()
            # lock the thread while updating the results
            self.lock.acquire()
            self.screenshot = screenshot
            self.lock.release()
            time.sleep(0.050)
