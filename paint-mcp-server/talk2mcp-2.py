import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
import google.generativeai as genai
from concurrent.futures import TimeoutError
from functools import partial
import traceback
import time
import win32gui
import win32con
import win32com.client
import win32api
import win32process
import psutil
from pywinauto import Application
from mcp.types import TextContent
import argparse
import logging
from datetime import datetime

# Model configuration
GEMINI_MODEL = "gemini-2.0-flash"

# Configure logging for client
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
client_log_file = os.path.join(log_dir, f"client_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
#    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(client_log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
logger.info(f"API Key loaded: {'Yes' if api_key else 'No'}")  # Will log Yes/No without exposing the key
genai.configure(api_key=api_key)

max_iterations = 3
last_response = None
iteration = 0
iteration_response = []

# Global variable for email flag
send_email_flag = False

async def generate_with_timeout(client, prompt, timeout=30):
    """Generate content with a timeout"""
    logger.info("Starting LLM generation...")
    logger.debug(f"LLM Generation Parameters - Model: {GEMINI_MODEL}, Timeout: {timeout}s")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    start_time = time.time()
    try:
        # Convert the synchronous generate_content call to run in a thread
        loop = asyncio.get_event_loop()
        
        # Use the configured Gemini model
        model = genai.GenerativeModel(GEMINI_MODEL)
        logger.debug(f"Initialized Gemini model: {GEMINI_MODEL}")
        
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: model.generate_content(contents=prompt)
            ),
            timeout=timeout
        )
        
        generation_time = time.time() - start_time
        logger.info(f"LLM generation completed successfully in {generation_time:.2f} seconds")
        logger.debug(f"Response length: {len(response.text) if response.text else 0} characters")
        return response
    except TimeoutError:
        generation_time = time.time() - start_time
        logger.error(f"LLM generation timed out after {generation_time:.2f} seconds!")
        raise
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"Error in LLM generation after {generation_time:.2f} seconds: {e}")
        # If first attempt fails, try with the configured model again
        try:
            logger.info(f"Retrying with {GEMINI_MODEL}...")
            model = genai.GenerativeModel(GEMINI_MODEL) 
            retry_start_time = time.time()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    lambda: model.generate_content(contents=prompt)
                ),
                timeout=timeout
            )
            retry_time = time.time() - retry_start_time
            logger.info(f"LLM generation completed with retry in {retry_time:.2f} seconds")
            return response
        except Exception as e2:
            retry_time = time.time() - retry_start_time
            logger.error(f"Error with retry model after {retry_time:.2f} seconds: {e2}")
            raise

def reset_state():
    """Reset all global variables to their initial state"""
    global last_response, iteration, iteration_response
    logger.debug("Resetting global state variables")
    logger.debug(f"Previous state - Iteration: {iteration}, Last response: {last_response is not None}, Response count: {len(iteration_response)}")
    last_response = None
    iteration = 0
    iteration_response = []
    logger.debug("State reset completed")

async def main(send_email=False):
    global send_email_flag
    send_email_flag = send_email
    
    logger.info("=== AI AGENT EXECUTION STARTED ===")
    logger.info(f"Email flag set to: {send_email}")
    execution_start_time = time.time()
    
    reset_state()  # Reset at the start of main
    logger.info("Starting main execution...")
    try:
        # Create a single MCP server connection
        logger.info("Establishing connection to MCP server...")
        server_params = StdioServerParameters(
            command="python",
            args=["example2.py"]
        )
        logger.debug(f"Server parameters: command='{server_params.command}', args={server_params.args}")

        async with stdio_client(server_params) as (read, write):
            logger.info("Connection established, creating session...")
            async with ClientSession(read, write) as session:
                logger.info("Session created, initializing...")
                await session.initialize()
                logger.debug("Session initialization completed")
                
                # Get available tools
                logger.info("Requesting tool list...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                logger.info(f"Successfully retrieved {len(tools)} tools")
                logger.debug(f"Available tools: {[tool.name for tool in tools]}")

                # Create system prompt with available tools
                logger.info("Creating system prompt...")
                logger.debug(f"Number of tools: {len(tools)}")
                
                try:
                    # First, let's inspect what a tool object looks like
                    # if tools:
                    #     logger.debug(f"First tool properties: {dir(tools[0])}")
                    #     logger.debug(f"First tool example: {tools[0]}")
                    
                    tools_description = []
                    for i, tool in enumerate(tools):
                        try:
                            # Get tool properties
                            params = tool.inputSchema
                            desc = getattr(tool, 'description', 'No description available')
                            name = getattr(tool, 'name', f'tool_{i}')
                            
                            # Format the input schema in a more readable way
                            if 'properties' in params:
                                param_details = []
                                for param_name, param_info in params['properties'].items():
                                    param_type = param_info.get('type', 'unknown')
                                    param_details.append(f"{param_name}: {param_type}")
                                params_str = ', '.join(param_details)
                            else:
                                params_str = 'no parameters'

                            tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                            tools_description.append(tool_desc)
                            logger.debug(f"Added description for tool: {tool_desc}")
                        except Exception as e:
                            logger.error(f"Error processing tool {i}: {e}")
                            tools_description.append(f"{i+1}. Error processing tool")
                    
                    tools_description = "\n".join(tools_description)
                    logger.info("Successfully created tools description")
                    logger.debug(f"Tools description length: {len(tools_description)} characters")
                except Exception as e:
                    logger.error(f"Error creating tools description: {e}")
                    tools_description = "Error loading tools"
                
                logger.info("Created system prompt...")
                
                system_prompt = f"""You are an AI agent that solves problems and visualizes results in Microsoft Paint. You have access to various tools for calculations and visualization.

Available tools:
{tools_description}

You must respond with EXACTLY ONE line in one of these formats (no additional text):
1. For function calls:
   FUNCTION_CALL: function_name|param1|param2|...
   
2. For visualization:
   FUNCTION_CALL: open_paint
   FUNCTION_CALL: draw_rectangle|x1|y1|x2|y2
   FUNCTION_CALL: add_text_in_paint|text

3. For final answers:
   FINAL_ANSWER: [your_answer]

Important:
- When a function returns multiple values, you need to process all of them
- Only give FINAL_ANSWER when you have completed all necessary calculations
- Do not repeat function calls with the same parameters  
- When solving a problem that needs visualization:
  1. First perform all calculations
  2. Then call open_paint (for visualization)
  3. Then draw_rectangle for the frame (for visualization)
  4. Finally add_text_in_paint with the result (for visualization)

Examples:
- FUNCTION_CALL: add|5|3
- FUNCTION_CALL: strings_to_chars_to_int|INDIA
- FUNCTION_CALL: open_paint
- FUNCTION_CALL: draw_rectangle|400|300|1200|600        # Filled black rectangle
- FUNCTION_CALL: add_text_in_paint|Result = 42          # Black text at (500,400)
"""

                query = """Find the ASCII values of characters in INDIA, calculate the sum of exponentials of those values, and visualize the result in Paint."""
                logger.info(f"Query defined: {query}")
                logger.info("Starting iteration loop...")
                
                # Use global iteration variables
                global iteration, last_response
                
                while iteration < max_iterations:
                    iteration_start_time = time.time()
                    logger.info("=" * 50)
                    logger.info(f"--- Iteration {iteration + 1} of {max_iterations} ---")
                    logger.info("=" * 50)
                    logger.debug(f"Iteration start time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                    
                    if last_response is None:
                        current_query = query
                        logger.debug("Using initial query (first iteration)")
                    else:
                        current_query = current_query + "\n\n" + " ".join(iteration_response)
                        current_query = current_query + "  What should I do next?"
                        logger.debug(f"Using updated query with {len(iteration_response)} previous responses")

                    # Get model's response with timeout
                    logger.info("Preparing to generate LLM response...")
                    prompt = f"{system_prompt}\n\nQuery: {current_query}"
                    logger.debug(f"Prompt length: {len(prompt)} characters")
                    try:
                        response = await generate_with_timeout(None, prompt)
                        response_text = response.text.strip()
                        logger.info(f"LLM Response received: {response_text}")
                        logger.debug(f"Response processing time: {time.time() - iteration_start_time:.2f} seconds")
                        
                        # Find the FUNCTION_CALL line in the response
                        for line in response_text.split('\n'):
                            line = line.strip()
                            if line.startswith("FUNCTION_CALL:"):
                                response_text = line
                                break
                        
                        if response_text.startswith("FUNCTION_CALL:"):
                            _, function_info = response_text.split(":", 1)
                            parts = [p.strip() for p in function_info.split("|")]
                            func_name, params = parts[0], parts[1:]
                            
                            logger.info(f"Processing function call: {func_name}")
                            logger.debug(f"Raw function info: {function_info}")
                            logger.debug(f"Split parts: {parts}")
                            logger.debug(f"Function name: {func_name}")
                            logger.debug(f"Raw parameters: {params}")
                            logger.debug(f"Parameter count: {len(params)}")
                            
                            try:
                                # Find the matching tool to get its input schema
                                tool = next((t for t in tools if t.name == func_name), None)
                                if not tool:
                                    logger.error(f"Unknown tool: {func_name}")
                                    logger.debug(f"Available tools: {[t.name for t in tools]}")
                                    raise ValueError(f"Unknown tool: {func_name}")

                                logger.info(f"Found tool: {tool.name}")
                                logger.debug(f"Tool schema: {tool.inputSchema}")

                                # Prepare arguments according to the tool's input schema
                                arguments = {}
                                schema_properties = tool.inputSchema.get('properties', {})
                                logger.debug(f"Schema properties: {schema_properties}")

                                for param_name, param_info in schema_properties.items():
                                    if not params:  # Check if we have enough parameters
                                        raise ValueError(f"Not enough parameters provided for {func_name}")
                                        
                                    value = params.pop(0)  # Get and remove the first parameter
                                    param_type = param_info.get('type', 'string')
                                    
                                    logger.debug(f"Converting parameter {param_name} with value '{value}' to type {param_type}")
                                    
                                    # Convert the value to the correct type based on the schema
                                    if param_type == 'integer':
                                        arguments[param_name] = int(value)
                                    elif param_type == 'number':
                                        arguments[param_name] = float(value)
                                    elif param_type == 'array':
                                        # Handle array input - convert all remaining parameters to integers
                                        if func_name == "int_list_to_exponential_sum":
                                            # For int_list_to_exponential_sum, check if the value is a comma-separated string
                                            if isinstance(value, str) and ',' in value:
                                                # Split the comma-separated string and convert to integers
                                                array_values = [int(x.strip()) for x in value.split(',')]
                                            else:
                                                # Use all parameters including the first one
                                                array_values = [int(value)] + [int(p.strip()) for p in params]
                                            arguments[param_name] = array_values
                                            # Clear the params list since we've used all values
                                            params.clear()
                                        else:
                                            # For other array parameters, handle as before
                                            if isinstance(value, str):
                                                value = value.strip('[]').split(',')
                                            arguments[param_name] = [int(x.strip()) for x in value]
                                    else:
                                        arguments[param_name] = str(value)

                                logger.info(f"Final arguments prepared: {arguments}")
                                logger.info(f"Calling tool {func_name} with arguments: {arguments}")
                                
                                tool_call_start_time = time.time()
                                result = await session.call_tool(func_name, arguments=arguments)
                                tool_call_time = time.time() - tool_call_start_time
                                logger.info(f"Tool call completed in {tool_call_time:.2f} seconds")
                                logger.debug(f"Raw result: {result}")
                                
                                # Get the full result content
                                if hasattr(result, 'content'):
                                    logger.debug("Result has content attribute")
                                    # Handle multiple content items
                                    if isinstance(result.content, list):
                                        iteration_result = [
                                            item.text if hasattr(item, 'text') else str(item)
                                            for item in result.content
                                        ]
                                    else:
                                        iteration_result = str(result.content)
                                else:
                                    logger.debug("Result has no content attribute")
                                    iteration_result = str(result)
                                    
                                logger.info(f"Final iteration result: {iteration_result}")
                                
                                # Format the response based on result type
                                if isinstance(iteration_result, list):
                                    result_str = f"[{', '.join(iteration_result)}]"
                                else:
                                    result_str = str(iteration_result)
                                
                                logger.info(f"Function {func_name} returned: {result_str}")
                                logger.debug(f"Function call summary - Name: {func_name}, Args: {arguments}, Result: {result_str}")
                                
                                iteration_response.append(
                                    f"In the {iteration + 1} iteration you called {func_name} with {arguments} parameters, "
                                    f"and the function returned {result_str}."
                                )
                                last_response = iteration_result

                                # If we've completed the calculation, proceed with visualization
                                if func_name == "int_list_to_exponential_sum":
                                    logger.info("")
                                    logger.info("=== AI AGENT EXECUTION (CALCULATION) COMPLETE, PROCEEDING WITH VISUALIZATION ===")
                                    visualization_start_time = time.time()
                                    logger.info(f"Visualization started at: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
 
                                    logger.info("Step 1: Opening Microsoft Paint...")
                                    # Open Paint
                                    paint_open_start = time.time()
                                    result = await session.call_tool("open_paint")
                                    paint_open_time = time.time() - paint_open_start
                                    logger.info(f"[SUCCESS] Paint opened successfully in {paint_open_time:.2f} seconds")
                                    logger.info(f"Paint open result: {result.content[0].text}")
                                    await asyncio.sleep(1)

                                    logger.info("Step 2: Drawing rectangle frame...")
                                    # Draw rectangle
                                    rect_args = {"x1": 400, "y1": 300, "x2": 1200, "y2": 600}
                                    logger.debug(f"Rectangle drawing parameters: {rect_args}")
                                    rect_draw_start = time.time()
                                    result = await session.call_tool("draw_rectangle", arguments=rect_args)
                                    rect_draw_time = time.time() - rect_draw_start
                                    logger.info(f"[SUCCESS] Rectangle drawn successfully in {rect_draw_time:.2f} seconds")
                                    logger.info(f"Rectangle draw result: {result.content[0].text}")

                                    logger.info("Step 3: Adding result text...")
                                    # Add text with the result
                                    text_args = {"text": f"Result = {result_str}"}
                                    logger.debug(f"Text addition parameters: {text_args}")
                                    text_add_start = time.time()
                                    result = await session.call_tool("add_text_in_paint", arguments=text_args)
                                    text_add_time = time.time() - text_add_start
                                    logger.info(f"[SUCCESS] Text added successfully in {text_add_time:.2f} seconds")
                                    logger.info(f"Text add result: {result.content[0].text}")
                                    
                                    total_viz_time = time.time() - visualization_start_time
                                    logger.info("=== VISUALIZATION COMPLETE ===")
                                    logger.info(f"Total visualization time: {total_viz_time:.2f} seconds")
                                    logger.info("The result has been displayed in Microsoft Paint.")
                                    logger.info("You can find the visualization in the Paint window.")
                                    
                                    # Log iteration completion before breaking
                                    iteration_end_time = time.time()
                                    iteration_duration = iteration_end_time - iteration_start_time
                                    logger.info("-" * 30)
                                    logger.info(f"Iteration {iteration + 1} completed in {iteration_duration:.2f} seconds")
                                    logger.info("-" * 30)
                                    break

                            except Exception as e:
                                logger.error(f"Error in tool execution: {str(e)}")
                                logger.error(f"Error type: {type(e)}")
                                logger.debug("Full traceback:", exc_info=True)
                                iteration_response.append(f"Error in iteration {iteration + 1}: {str(e)}")
                                break

                        elif response_text.startswith("FINAL_ANSWER:"):
                            logger.info("=== AGENT EXECUTION COMPLETE ===")
                            break

                        iteration_end_time = time.time()
                        iteration_duration = iteration_end_time - iteration_start_time
                        logger.info("-" * 30)
                        logger.info(f"Iteration {iteration + 1} completed in {iteration_duration:.2f} seconds")
                        logger.info("-" * 30)
                        iteration += 1
                    except Exception as e:
                        logger.error(f"Failed to get LLM response: {e}")
                        logger.debug("LLM response error traceback:", exc_info=True)
                        break

                total_execution_time = time.time() - execution_start_time
                logger.info(f"=== AI AGENT EXECUTION COMPLETED ===")
                logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
                logger.info(f"Total iterations: {iteration + 1}")  # Add 1 since iteration is 0-based

    except Exception as e:
        total_execution_time = time.time() - execution_start_time
        logger.error(f"Error in main execution after {total_execution_time:.2f} seconds: {e}")
        logger.error("Main execution error traceback:", exc_info=True)
    finally:
        logger.info("Resetting state and cleaning up...")
        reset_state()  # Reset at the end of main

def find_paint_window():
    """Find Paint window using multiple methods"""
    def enum_windows_callback(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            # Paint window class is usually 'MSPaintApp' or contains 'Paint'
            if "Paint" in window_text or "MSPaintApp" in class_name:
                result.append(hwnd)
        return True

    paint_windows = []
    win32gui.EnumWindows(enum_windows_callback, paint_windows)
    return paint_windows[0] if paint_windows else None

def force_activate_window(hwnd):
    """Force activate a window using multiple methods"""
    if not hwnd:
        return False
    
    try:
        # Get current foreground window
        current_fg = win32gui.GetForegroundWindow()
        
        # Get the current window's thread
        current_thread = win32api.GetCurrentThreadId()
        
        # Get the target window's thread
        target_thread = win32process.GetWindowThreadProcessId(hwnd)[0]
        
        # Attach the threads
        win32process.AttachThreadInput(target_thread, current_thread, True)
        
        try:
            # Show the window
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            
            # Force foreground
            win32gui.SetForegroundWindow(hwnd)
            
            # Maximize
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            
            # Additional focus attempts
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetFocus(hwnd)
            
        finally:
            # Detach threads
            win32process.AttachThreadInput(target_thread, current_thread, False)
        
        # Wait for window to be active
        time.sleep(0.5)
        return win32gui.GetForegroundWindow() == hwnd
        
    except Exception as e:
        logger.error(f"Error activating window: {e}")
        return False

def ensure_paint_active():
    """Ensure Paint is running and active"""
    # Find Paint process
    paint_pid = None
    for proc in psutil.process_iter(['pid', 'name']):
        if 'mspaint' in proc.info['name'].lower():
            paint_pid = proc.info['pid']
            break
    
    if not paint_pid:
        logger.debug("Paint process not found")
        return False
    
    # Find Paint window
    hwnd = find_paint_window()
    if not hwnd:
        logger.debug("Paint window not found")
        return False
    
    # Force activate window
    success = True
    # success = force_activate_window(hwnd)
    # if success:
    #     logger.debug("Paint window successfully activated")
    # else:
    #     logger.debug("Failed to activate Paint window")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
