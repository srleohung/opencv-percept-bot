from bot import BotState, Bot
import cv2
from utils import detect_template_in_image 

class Bot(Bot):
    """
    A custom bot class that inherits from the Bot class,
    allowing for specific modifications to the run method.
    """
    def run(self):
        """
        The main loop of the custom bot, executing its logic based on the current state.
        This method overrides the base class's run method with customized behavior.
        """
        needle_img = None
        while not self.stopped:
            self.wait()  # Wait for a short period before each iteration
            
            # Custom logic can be added here
            if self.state == BotState.INITIALIZING:
                # Perform specific actions during initialization
                print("Custom Bot is initializing.")
                
                needle_img = cv2.imread('needle.jpg', cv2.IMREAD_UNCHANGED)
                
                self.lock.acquire()
                self.state = BotState.SEARCHING
                self.lock.release()

            elif self.state == BotState.SEARCHING:
                # Custom searching logic
                print("Custom Bot is searching for a target.")
                
                # Call your custom template detection logic here
                self.rectangles = detect_template_in_image(self.screenshot, template=needle_img)

                # If any template is found, transition to the TRADING state
                if len(self.rectangles) > 0:
                    self.lock.acquire()
                    self.state = BotState.TRADING
                    self.lock.release()

            elif self.state == BotState.TRADING:
                # Custom trading logic
                print("Custom Bot is executing trading actions.")
                
                # Execute your custom trading logic here
                self.lock.acquire()
                self.state = BotState.BACKTRACKING
                self.lock.release()

            elif self.state == BotState.BACKTRACKING:
                # Custom backtracking logic
                print("Custom Bot is backtracking to search.")
                
                self.lock.acquire()
                self.state = BotState.SEARCHING
                self.lock.release()