import cv2
import pyautogui
import time
from threading import Thread, Lock
from utils import extract_text_from_image, get_rectangle_from_points, detect_template_in_image 

class BotState:
    """
    Enumeration to represent the different states of the bot.
    
    Attributes:
        INITIALIZING (int): Initial state of the bot before starting.
        SEARCHING (int): State when the bot is searching for a target.
        TRADING (int): State when the bot is performing a trade action.
        BACKTRACKING (int): State when the bot is resetting and preparing to search again.
    """
    INITIALIZING = 0
    SEARCHING = 1
    TRADING = 2 
    BACKTRACKING = 3

class Bot:
    """
    A bot class for automating game actions using screen detection and automation tools.

    The bot performs actions in a loop, switching between different states (INITIALIZING, SEARCHING, TRADING, BACKTRACKING),
    based on its current environment, detected via screenshots and template matching.

    Attributes:
        stopped (bool): Flag to control the main loop of the bot.
        lock (threading.Lock): A lock to handle shared resources (e.g., screenshots) safely.
        state (BotState): Current state of the bot.
        screenshot (ndarray): Latest screenshot of the screen.
        rectangles (list): List of rectangles where detected templates are found.
    """

    # **************************************************
    # * Threading Properties
    # **************************************************
    stopped = True  # Control the bot's running state
    lock = None  # Lock to manage concurrent access to shared resources
    state = None  # Current state of the bot
    screenshot = None  # Latest screenshot data
    rectangles = None  # List of detected rectangles on the screenshot

    def __init__(self):
        """
        Initializes the bot with a default state of INITIALIZING.
        """
        # Create a thread lock object for safe multi-threaded access
        self.lock = Lock()
        self.state = BotState.INITIALIZING

    # **************************************************
    # * Utility Functions
    # **************************************************

    def extract_text_from_area(self, points):
        """
        Extract textual content from a specific region in the screenshot.

        Args:
            points (tuple): A tuple containing two points (top-left and bottom-right) that define the area of the screenshot.

        Returns:
            str: The extracted text from the image in the specified region.
        """
        content = extract_text_from_image(self.screenshot, points[0], points[1])
        self.rectangles = [get_rectangle_from_points(points[0], points[1])]
        return content

    def extract_integer_from_area(self, points):
        """
        Extract an integer value from a specific region in the screenshot.

        Args:
            points (tuple): A tuple containing two points (top-left and bottom-right) that define the area of the screenshot.

        Returns:
            int: The extracted integer from the image in the specified region.
                 Returns 0 if no valid integer is found.
        """
        content = extract_text_from_image(self.screenshot, points[0], points[1])
        self.rectangles = [get_rectangle_from_points(points[0], points[1])]
        number_string = content.replace(',', '').replace(' ', '').replace('.', '')
        try:
            return int(number_string)
        except ValueError:
            print(f'Integer not found in range, content: {number_string}')
            return 0

    # **************************************************
    # * Action Functions
    # **************************************************

    def click(self, point, delay=0.100):
        """
        Simulate a mouse click at a given point on the screen.

        Args:
            point (tuple): A tuple representing the (x, y) coordinates of the point to click.
            delay (float, optional): Time to wait before clicking (default is 0.100 seconds).
        """
        pyautogui.moveTo(x=point[0], y=point[1])
        time.sleep(delay)
        pyautogui.click()

    def wait(self, seconds=1.000):
        """
        Pause execution for a given number of seconds.

        Args:
            seconds (float, optional): The number of seconds to wait (default is 1.000 seconds).
        """
        time.sleep(seconds)

    # **************************************************
    # * Threading Methods
    # **************************************************

    def update_screenshot(self, screenshot):
        """
        Update the current screenshot safely in a multi-threaded environment.

        Args:
            screenshot (ndarray): The new screenshot to update.
        """
        self.lock.acquire()
        self.screenshot = screenshot
        self.lock.release()

    def start(self):
        """
        Start the bot in a separate thread.
        """
        self.stopped = False
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        """
        Stop the bot's operation.
        """
        self.stopped = True

    # Main logic controller
    def run(self):
        """
        The main loop of the bot, executing its logic based on the current state.
        
        Handles state transitions and performs actions based on the bot's state,
        such as searching for a template, executing trades, and backtracking.
        """
        needle_img = None
        while not self.stopped:
            self.wait()  # Wait for a short period before each iteration

            if self.state == BotState.INITIALIZING:
                # Prepare the needle image before starting the search
                needle_img = cv2.imread('needle.jpg', cv2.IMREAD_UNCHANGED)

                # Transition to the SEARCHING state
                self.lock.acquire()
                self.state = BotState.SEARCHING
                self.lock.release()

            elif self.state == BotState.SEARCHING:
                # Detect the template in the current screenshot
                self.rectangles = detect_template_in_image(self.screenshot, template=needle_img)

                # If any template is found, transition to the TRADING state
                if len(self.rectangles) > 0:
                    self.lock.acquire()
                    self.state = BotState.TRADING
                    self.lock.release()

            elif self.state == BotState.TRADING:
                # Execute trading logic and then backtrack
                self.lock.acquire()
                self.state = BotState.BACKTRACKING
                self.lock.release()

            elif self.state == BotState.BACKTRACKING:
                # Reset to searching after backtracking
                self.lock.acquire()
                self.state = BotState.SEARCHING
                self.lock.release()