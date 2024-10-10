import cv2 
import pyautogui
import time
from threading import Thread, Lock
import pytesseract
import numpy as np

class Bot:

    # **************************************************
    # * Threading Properties
    # **************************************************
    stopped = True
    lock = None
    screenshot = None

    # **************************************************
    # * Properties
    # **************************************************
    POSITION_CLOSE_BUTTON = (513, 193)
    POSITION_BUY_BUTTON = (513, 193)
    AREA_PRICE_CONTENT = [(274, 658), (392, 683)]

    def __init__(self):
        # create a thread lock object
        self.lock = Lock()

    # **************************************************
    # * Utility Functions
    # **************************************************
    def image_to_string(self, start, end):
        p1_x, p1_y = start
        p2_x, p2_y = end
        if start[0] > end[0]:
            p1_x = end[0]
            p2_x = start[0]
        if start[1] > end[1]:
            p1_y = end[1]
            p2_y = start[1]
        cropped_screenshot = self.screenshot[p1_y:p2_y, p1_x:p2_x]
        content = pytesseract.image_to_string(cropped_screenshot)
        return content

    def get_integer_from_image(self, points):
        content = self.image_to_string(points[0], points[1])
        try:
            return int(content)
        except ValueError:
            print("Price not found in range")
            return 0
        
    def find_in_image(self, screenshot, needle_img=None, needle_img_path=None, threshold=0.8):
        
        if needle_img is None:
            needle_img = cv2.imread(needle_img_path, cv2.IMREAD_UNCHANGED)

        # run the OpenCV algorithm
        result = cv2.matchTemplate(screenshot, needle_img, self.method)

        # Get the all the positions from the match result that exceed our threshold
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        # You'll notice a lot of overlapping rectangles get drawn. We can eliminate those redundant
        # locations by using groupRectangles().
        # First we need to create the list of [x, y, w, h] rectangles
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            # Add every box to the list twice in order to retain single (non-overlapping) boxes
            rectangles.append(rect)
            rectangles.append(rect)
        # Apply group rectangles.
        # The groupThreshold parameter should usually be 1. If you put it at 0 then no grouping is
        # done. If you put it at 2 then an object needs at least 3 overlapping rectangles to appear
        # in the result. I've set eps to 0.5, which is:
        # "Relative difference between sides of the rectangles to merge them into a group."
        rectangles, weights = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

        return rectangles

    # **************************************************
    # * Action Functions
    # **************************************************
    def click(self, point, delay=0.500):
        pyautogui.moveTo(x=point[0], y=point[1])
        time.sleep(delay)
        pyautogui.click()

    # **************************************************
    # * Threading Methods
    # **************************************************
    def update_screenshot(self, screenshot):
        self.lock.acquire()
        self.screenshot = screenshot
        self.lock.release()

    def start(self):
        self.stopped = False
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        self.stopped = True

    # main logic controller
    def run(self):
        while not self.stopped:
            self.click(self.POSITION_BUY_BUTTON)
            price = self.get_integer_from_image(self.AREA_PRICE_CONTENT)
            if price > 0 and price < 10000:
                self.click(self.POSITION_BUY_BUTTON)
            self.click(self.POSITION_CLOSE_BUTTON)




