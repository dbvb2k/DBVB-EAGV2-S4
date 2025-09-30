# basic import 
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp import types
from PIL import Image as PILImage
import math
import sys
from pywinauto.application import Application
import win32gui
import win32con
import time
from win32api import GetSystemMetrics
import logging
import os
from datetime import datetime
from dotenv import load_dotenv



# Load environment variables
load_dotenv()

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"debug_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# instantiate an MCP server client
mcp = FastMCP("Calculator")

# DEFINE TOOLS

#addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    logger.info(f"CALLED: add(a: int, b: int) -> int: a={a}, b={b}")
    result = int(a + b)
    logger.info(f"RETURNED: add -> {result}")
    return result

@mcp.tool()
def add_list(l: list) -> int:
    """Add all numbers in a list"""
    logger.info(f"CALLED: add_list(l: list) -> int: l={l}")
    result = sum(l)
    logger.info(f"RETURNED: add_list -> {result}")
    return result

# subtraction tool
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    logger.info(f"CALLED: subtract(a: int, b: int) -> int: a={a}, b={b}")
    result = int(a - b)
    logger.info(f"RETURNED: subtract -> {result}")
    return result

# multiplication tool
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    logger.info(f"CALLED: multiply(a: int, b: int) -> int: a={a}, b={b}")
    result = int(a * b)
    logger.info(f"RETURNED: multiply -> {result}")
    return result

#  division tool
@mcp.tool() 
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    logger.info(f"CALLED: divide(a: int, b: int) -> float: a={a}, b={b}")
    result = float(a / b)
    logger.info(f"RETURNED: divide -> {result}")
    return result

# power tool
@mcp.tool()
def power(a: int, b: int) -> int:
    """Power of two numbers"""
    logger.info(f"CALLED: power(a: int, b: int) -> int: a={a}, b={b}")
    result = int(a ** b)
    logger.info(f"RETURNED: power -> {result}")
    return result

# square root tool
@mcp.tool()
def sqrt(a: int) -> float:
    """Square root of a number"""
    logger.info(f"CALLED: sqrt(a: int) -> float: a={a}")
    result = float(a ** 0.5)
    logger.info(f"RETURNED: sqrt -> {result}")
    return result

# cube root tool
@mcp.tool()
def cbrt(a: int) -> float:
    """Cube root of a number"""
    logger.info(f"CALLED: cbrt(a: int) -> float: a={a}")
    result = float(a ** (1/3))
    logger.info(f"RETURNED: cbrt -> {result}")
    return result

# factorial tool
@mcp.tool()
def factorial(a: int) -> int:
    """factorial of a number"""
    logger.info(f"CALLED: factorial(a: int) -> int: a={a}")
    result = int(math.factorial(a))
    logger.info(f"RETURNED: factorial -> {result}")
    return result

# log tool
@mcp.tool()
def log(a: int) -> float:
    """log of a number"""
    logger.info(f"CALLED: log(a: int) -> float: a={a}")
    result = float(math.log(a))
    logger.info(f"RETURNED: log -> {result}")
    return result

# remainder tool
@mcp.tool()
def remainder(a: int, b: int) -> int:
    """remainder of two numbers divison"""
    logger.info(f"CALLED: remainder(a: int, b: int) -> int: a={a}, b={b}")
    result = int(a % b)
    logger.info(f"RETURNED: remainder -> {result}")
    return result

# sin tool
@mcp.tool()
def sin(a: int) -> float:
    """sin of a number"""
    logger.info(f"CALLED: sin(a: int) -> float: a={a}")
    result = float(math.sin(a))
    logger.info(f"RETURNED: sin -> {result}")
    return result

# cos tool
@mcp.tool()
def cos(a: int) -> float:
    """cos of a number"""
    logger.info(f"CALLED: cos(a: int) -> float: a={a}")
    result = float(math.cos(a))
    logger.info(f"RETURNED: cos -> {result}")
    return result

# tan tool
@mcp.tool()
def tan(a: int) -> float:
    """tan of a number"""
    logger.info(f"CALLED: tan(a: int) -> float: a={a}")
    result = float(math.tan(a))
    logger.info(f"RETURNED: tan -> {result}")
    return result

# mine tool
@mcp.tool()
def mine(a: int, b: int) -> int:
    """special mining tool"""
    logger.info(f"CALLED: mine(a: int, b: int) -> int: a={a}, b={b}")
    result = int(a - b - b)
    logger.info(f"RETURNED: mine -> {result}")
    return result

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    logger.info(f"CALLED: create_thumbnail(image_path: str) -> Image: image_path='{image_path}'")
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    result = Image(data=img.tobytes(), format="png")
    logger.info(f"RETURNED: create_thumbnail -> Image object created")
    return result

@mcp.tool()
def strings_to_chars_to_int(string: str) -> list[int]:
    """Return the ASCII values of the characters in a word"""
    logger.info(f"CALLED: strings_to_chars_to_int(string: str) -> list[int]: string='{string}'")
    result = [int(ord(char)) for char in string]
    logger.info(f"RETURNED: strings_to_chars_to_int -> {result}")
    return result

@mcp.tool()
def int_list_to_exponential_sum(int_list: list) -> float:
    """Return sum of exponentials of numbers in a list"""
    logger.info(f"CALLED: int_list_to_exponential_sum(int_list: list) -> float: int_list={int_list}")
    result = sum(math.exp(i) for i in int_list)
    logger.info(f"RETURNED: int_list_to_exponential_sum -> {result}")
    return result

@mcp.tool()
def fibonacci_numbers(n: int) -> list:
    """Return the first n Fibonacci Numbers"""
    logger.info(f"CALLED: fibonacci_numbers(n: int) -> list: n={n}")
    if n <= 0:
        result = []
        logger.info(f"RETURNED: fibonacci_numbers -> {result} (n <= 0)")
        return result
    fib_sequence = [0, 1]
    for _ in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    result = fib_sequence[:n]
    logger.info(f"RETURNED: fibonacci_numbers -> {result}")
    return result


@mcp.tool()
async def draw_rectangle(x1: int, y1: int, x2: int, y2: int) -> dict:
    """Draw a rectangle in Paint from (x1,y1) to (x2,y2)"""
    global paint_app
    logger.info(f"CALLED: draw_rectangle(x1: int, y1: int, x2: int, y2: int) -> dict: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    try:
        if not paint_app:
            error_msg = "Paint is not open. Please call open_paint first."
            logger.error(f"RETURNED: draw_rectangle -> ERROR: {error_msg}")
            return {"content": [TextContent(type="text", text=error_msg)]}
        
        logger.debug("Starting rectangle drawing operation")
        paint_window = paint_app.window(class_name='MSPaintApp')
        
        # Ensure Paint window is active
        if not paint_window.has_focus():
            logger.debug("Paint window doesn't have focus, setting focus...")
            paint_window.set_focus()
            time.sleep(1)
            
        # Click Rectangle tool
        logger.debug("Selecting rectangle tool at coordinates (445, 70)")
        paint_window.click_input(coords=(445, 70))
        time.sleep(1)
        
        # Get canvas area
        canvas = paint_window.child_window(class_name='MSPaintView')
        
        # Log canvas position and size for debugging
        canvas_rect = canvas.rectangle()
        logger.debug(f"Canvas Rectangle: {canvas_rect}")
        
        # Draw rectangle with logging
        logger.debug(f"Drawing rectangle from ({x1},{y1}) to ({x2},{y2})")

        # Draw rectangle - coordinates should be relative to the Paint window
        logger.debug(f"Clicking at: ({x1}, {y1})")
        canvas.click_input(coords=(x1, y1))
        time.sleep(1)

        logger.debug(f"Pressing mouse at: ({x1}, {y1})")
        canvas.press_mouse_input(coords=(x1, y1))
        time.sleep(1)

        logger.debug(f"Releasing mouse at: ({x2}, {y2})")
        canvas.release_mouse_input(coords=(x2, y2))
        time.sleep(1)  

        # logger.debug(f"Clicking at: ({x2}, {y2+40})")
        # canvas.click_input(coords=(x2, y2+40))
        # time.sleep(1)      

        result_text = f"Rectangle drawn from ({x1},{y1}) to ({x2},{y2})"
        logger.info(f"RETURNED: draw_rectangle -> {result_text}")
        return {
            "content": [TextContent(type="text", text=result_text)]
        }
    except Exception as e:
        error_msg = f"Error drawing rectangle: {str(e)}"
        logger.error(f"RETURNED: draw_rectangle -> ERROR: {error_msg}")
        logger.error(f"Error in draw_rectangle: {str(e)}")
        return {"content": [TextContent(type="text", text=error_msg)]}

@mcp.tool()
async def add_text_in_paint(text: str) -> dict:
    """Add text in Paint"""
    global paint_app
    logger.info(f"CALLED: add_text_in_paint(text: str) -> dict: text='{text}'")
    try:
        if not paint_app:
            error_msg = "Paint is not open. Please call open_paint first."
            logger.error(f"RETURNED: add_text_in_paint -> ERROR: {error_msg}")
            return {"content": [TextContent(type="text", text=error_msg)]}
        
        logger.debug("Starting text addition operation")
        paint_window = paint_app.window(class_name='MSPaintApp')
        
        # Ensure Paint window is active
        if not paint_window.has_focus():
            logger.debug("Paint window doesn't have focus, setting focus...")
            paint_window.set_focus()
            time.sleep(0.5)
        
        # Select green color
        logger.debug("Selecting green color at coordinates (895, 61)")
        paint_window.click_input(coords=(895, 61))
        time.sleep(0.5)
        
        # Select Text tool
        logger.debug("Selecting text tool at coordinates (290, 70)")
        paint_window.click_input(coords=(290, 70))
        time.sleep(0.5)
        
        # Get canvas
        canvas = paint_window.child_window(class_name='MSPaintView')
        
        # Click to start typing (inside rectangle)
        text_x, text_y = 500, 300  # Adjusted coordinates
        logger.debug(f"Clicking for text at ({text_x}, {text_y})")
        canvas.click_input(coords=(text_x, text_y))
        time.sleep(0.5)
        
        # Type text
        logger.debug(f"Typing text: '{text}'")
        paint_window.type_keys(text, with_spaces=True)
        time.sleep(0.5)
        
        # Click outside to finish
        logger.debug("Clicking outside to finish text input")
        canvas.click_input(coords=(50, 50))
        time.sleep(0.5)

        result_text = f"Text:'{text}' added at ({text_x},{text_y})"
        logger.info(f"RETURNED: add_text_in_paint -> {result_text}")
        return {
            "content": [TextContent(type="text", text=result_text)]
        }
    except Exception as e:
        error_msg = f"Error adding text: {str(e)}"
        logger.error(f"RETURNED: add_text_in_paint -> ERROR: {error_msg}")
        logger.error(f"Error in add_text_in_paint: {str(e)}")
        return {"content": [TextContent(type="text", text=error_msg)]}

@mcp.tool()
async def open_paint() -> dict:
    """Open Microsoft Paint maximized"""
    global paint_app
    logger.info("CALLED: open_paint() -> dict")
    try:
        logger.debug("Starting Paint opening operation")
        paint_app = Application().start('mspaint.exe')
        time.sleep(1)
        logger.debug("Paint application started successfully")
        
        paint_window = paint_app.window(class_name='MSPaintApp')
        
        # Get initial window position
        initial_rect = paint_window.rectangle()
        logger.debug(f"Initial Paint window rectangle: {initial_rect}")
        
        # Maximize window
        logger.debug("Maximizing Paint window")
        win32gui.ShowWindow(paint_window.handle, win32con.SW_MAXIMIZE)
        time.sleep(0.5)
        
        # Get maximized position
        max_rect = paint_window.rectangle()
        logger.debug(f"Maximized Paint window rectangle: {max_rect}")
        
        # Get canvas
        canvas = paint_window.child_window(class_name='MSPaintView')
        canvas_rect = canvas.rectangle()
        logger.debug(f"Canvas rectangle: {canvas_rect}")
        
        result_text = "Paint opened successfully and maximized"
        logger.info(f"RETURNED: open_paint -> {result_text}")
        return {
            "content": [TextContent(type="text", text=result_text)]
        }
    except Exception as e:
        error_msg = f"Error opening Paint: {str(e)}"
        logger.error(f"RETURNED: open_paint -> ERROR: {error_msg}")
        logger.error(f"Error in open_paint: {str(e)}")
        return {"content": [TextContent(type="text", text=error_msg)]}


# DEFINE RESOURCES

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    logger.info(f"CALLED: get_greeting(name: str) -> str: name='{name}'")
    result = f"Hello, {name}!"
    logger.info(f"RETURNED: get_greeting -> {result}")
    return result


# DEFINE AVAILABLE PROMPTS
@mcp.prompt()
def review_code(code: str) -> str:
    logger.info(f"CALLED: review_code(code: str) -> str: code_length={len(code)}")
    result = f"Please review this code:\n\n{code}"
    logger.info(f"RETURNED: review_code -> generated prompt with {len(result)} characters")
    return result


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    logger.info(f"CALLED: debug_error(error: str) -> list[base.Message]: error='{error}'")
    result = [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]
    logger.info(f"RETURNED: debug_error -> generated {len(result)} messages")
    return result

system_prompt = f"""
...
Examples:
- FUNCTION_CALL: add|5|3
- FUNCTION_CALL: strings_to_chars_to_int|INDIA
- FUNCTION_CALL: open_paint
- FUNCTION_CALL: draw_rectangle|100|100|500|400  # Large visible rectangle
- FUNCTION_CALL: add_text_in_paint|Result = 42  # Text will be placed at (100,100)
...
"""

if __name__ == "__main__":
    # Check if running with mcp dev command
    logger.info("STARTING")
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run()  # Run without transport for dev server
    else:
        mcp.run(transport="stdio")  # Run with stdio for direct execution
