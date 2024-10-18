import discord
import asyncio
from threading import Thread
from io import BytesIO
from PIL import Image

class DiscordMessenger:
    """
    A Discord bot that runs in a separate thread and can send messages to a specified channel asynchronously.

    Attributes:
        channel_id (int): The Discord channel ID where the bot sends messages.
        token (str): The bot's token used for authentication with Discord.
        client (DiscordClient): The custom Discord client for interacting with the Discord API.
        loop (asyncio.AbstractEventLoop): The asyncio event loop for handling asynchronous tasks.
    """

    def __init__(self, channel_id="", token=""):
        """
        Initializes the DiscordBot with a specified channel ID and bot token.

        Args:
            channel_id (str or int): The Discord channel ID where the bot will send messages.
            token (str): The bot's token for authentication with Discord.
        """
        self.channel_id = int(channel_id)  # Ensure channel_id is an integer
        self.token = token
        self.client = DiscordClient(intents=discord.Intents.default())
        self.loop = asyncio.get_event_loop()

    def send(self, message, delete_after=None):
        """
        Sends a message to the specified Discord channel asynchronously.

        Args:
            message (str): The message content to send to the Discord channel.
        """
        # Schedule the coroutine to run in the client's event loop
        asyncio.run_coroutine_threadsafe(self.client.send_message(self.channel_id, message, delete_after=delete_after), self.client.loop)
    
    def send_image(self, image_bytes, filename="image.png", delete_after=None):
        """
        Sends an image to the specified Discord channel asynchronously.

        Args:
            image_bytes (bytes): The image data as bytes.
            filename (str): The filename to use when sending the image (default: "image.png").
        """
        asyncio.run_coroutine_threadsafe(self.client.send_image(self.channel_id, image_bytes, filename, delete_after=delete_after), self.client.loop)

    def send_screencap(self, screencap, top_left, bottom_right, filename="screencap.png", delete_after=None):
        """
        Converts a numpy array screencap to bytes and sends it to the Discord channel.

        Args:
            screencap (numpy.ndarray): The full screenshot as a numpy array.
            start_point (tuple): (x, y) coordinates of the top-left corner of the crop.
            end_point (tuple): (x, y) coordinates of the bottom-right corner of the crop.
            filename (str): The filename to use for the image (default is "screencap.png").
        """

        # Ensure coordinates are in correct order (top-left, bottom-right)
        x1, y1 = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
        x2, y2 = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])
        
        # Extract the sub-region from the screencap using numpy slicing
        cropped_screencap = screencap[y1:y2, x1:x2]

        # Convert the numpy array to an image
        image = Image.fromarray(cropped_screencap)

        # Convert the image to bytes using BytesIO
        image_bytes_io = BytesIO()
        image.save(image_bytes_io, format='PNG')  # Save as PNG to the BytesIO buffer
        image_bytes_io.seek(0)  # Go to the start of the BytesIO buffer

        # Send the image to the Discord channel
        asyncio.run_coroutine_threadsafe(self.client.send_image(self.channel_id, image_bytes_io.getvalue(), filename, delete_after=delete_after), self.client.loop)

    def start(self):
        """
        Starts the Discord bot in a separate thread and connects it to Discord.
        """
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        """
        Stops the Discord bot and closes the Discord client connection.
        """
        asyncio.run_coroutine_threadsafe(self.client.close(), self.client.loop)

    def run(self):
        """
        Runs the Discord bot by logging into Discord using the provided token.
        """
        self.client.run(self.token)


class DiscordClient(discord.Client):
    """
    A custom Discord client for handling bot events and sending messages.

    Inherits from discord.Client to interact with Discord's API.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DiscordClient by calling the parent class constructor.
        """
        super().__init__(*args, **kwargs)

    async def on_ready(self):
        """
        Event handler called when the bot successfully logs into Discord and is ready to use.
        """
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def send_message(self, channel_id, message, delete_after=None):
        """
        Sends a message to a specified channel after ensuring the bot is ready.

        Args:
            channel_id (int): The ID of the Discord channel where the message will be sent.
            message (str): The message content to send to the Discord channel.
        """
        await self.wait_until_ready()  # Wait until the bot is fully connected and ready
        channel = self.get_channel(channel_id)  # Get the channel by ID
        if channel:
            await channel.send(message, delete_after=delete_after)  # Send the message if the channel is found
        else:
            print(f"Channel ID {channel_id} not found.")
    
    async def send_image(self, channel_id, image_bytes, filename, delete_after=None):
        """
        Sends an image as bytes to a specified channel after ensuring the bot is ready.

        Args:
            channel_id (int): The ID of the Discord channel where the image will be sent.
            image_bytes (bytes): The image data to send.
            filename (str): The name of the image file (default is "image.png").
        """
        await self.wait_until_ready()  # Wait until the bot is fully connected and ready
        channel = self.get_channel(channel_id)  # Get the channel by ID
        if channel:
            image_file = discord.File(BytesIO(image_bytes), filename=filename)
            await channel.send(file=image_file, delete_after=delete_after)  # Send the image file to the channel
        else:
            print(f"Channel ID {channel_id} not found.")