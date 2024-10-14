# opencv-percept-bot

A Python bot that captures screenshots from a specified window or screen area and triggers actions based on the visual input. The bot can extract text from images using OCR and provides functionality for drawing rectangles on the screen.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Commands](#commands)
- [License](#license)

## Requirements

- Python 3.6 or higher
- `opencv-python`
- `pytesseract`
- `pyautogui`
- `argparse`
- `numpy`

## Installation

### Windows

1. **Install Python**: Make sure Python is installed on your system. You can download it from the official [Python website](https://www.python.org/downloads/).

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   ```bash
   venv\Scripts\activate
   ```

4. **Install Required Packages**:
   ```bash
   pip install opencv-python pytesseract pyautogui numpy
   ```

5. **Install Tesseract-OCR**: Download and install Tesseract-OCR from [this link](https://github.com/tesseract-ocr/tesseract). Make sure to note the installation path, as you'll need it later.

### macOS

1. **Install Python**: Ensure you have Python installed on your system. You can install it via Homebrew:
   ```bash
   brew install python
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```

4. **Install Required Packages**:
   ```bash
   pip install opencv-python pytesseract pyautogui numpy
   ```

5. **Install Tesseract-OCR**: You can install Tesseract using Homebrew:
   ```bash
   brew install tesseract
   ```

## Usage

1. **Activate the Virtual Environment**:
   - For Windows:
     ```bash
     venv\Scripts\activate
     ```
   - For macOS:
     ```bash
     source venv/bin/activate
     ```

2. **Run the Bot**:
   You can run the bot with various command-line arguments. Here's a basic example:
   ```bash
   python main.py
   # Run a Custom Bot: Specify the bot you want to use by name.
   python main.py --bot custom_bot
   # Capture a Specific Window: Run the bot targeting a specific application window by its title.
   python main.py --window_name "opencv-percept-bot" 
   # Capture a Specific Screen Area: Define a rectangular area of the screen to capture, using the format x,y,width,height.
   python main.py --window_rect "0,0,560,1060"
   # Run with Multiple Arguments: You can combine arguments to specify both the bot and the target window or area:
   python main.py --bot custom_bot --window_name "opencv-percept-bot" --debug true
   python main.py --bot custom_bot --window_rect "0,0,560,1060" --debug true
   ```

## Commands

- **list_window_names**: List the names of all currently active windows.
- **--window_name**: Specify the name of the window to capture. Leave blank to capture the entire screen.
- **--window_rect**: Specify the rectangle to capture as 'x,y,width,height'. Leave blank for the entire screen.
- **--debug**: Enable or disable debug mode. Use 'true' or 'false'.
- **--bot**: Specify the name of the bot to use. Leave blank to use the default bot.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## References

This project was inspired by and built upon concepts learned from the [OpenCV Tutorials by LearnCodeByGaming](https://github.com/learncodebygaming/opencv_tutorials). These tutorials offer a comprehensive introduction to using OpenCV for creating game bots and similar applications.
