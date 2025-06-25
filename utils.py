# utils.py

import logging
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def configure_logging():
    """
    Configures the logging settings for the application, including detailed information 
    such as the file name and line number of the log message.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def calculate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the total cost of using OpenAI models based on token usage.

    :param model: The OpenAI model used (e.g., 'GPT-4o', 'GPT-4o mini', 'OpenAI o3-mini')
    :param input_tokens: Number of input tokens processed
    :param cached_tokens: Number of cached input tokens
    :param output_tokens: Number of output tokens generated
    :return: Total cost in USD
    """
    pricing = {
        "gpt-4o": {"input": 2.50 / 1_000_000, "cached": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.150 / 1_000_000, "cached": 0.075 / 1_000_000, "output": 0.600 / 1_000_000},
        "o3-mini-2025-01-31": {"input": 1.10 / 1_000_000, "cached": 0.55 / 1_000_000, "output": 4.40 / 1_000_000},
        "EMBEDDING": {"input": 0.1 / 1_000_000, "output": 0.1 / 1_000_000},
        "gpt-4.1-mini": {"input": 0.40 / 1_000_000, "cached": 0.1 / 1_000_000, "output": 1.60 / 1_000_000},
        "gpt-4.1-nano": {"input": 0.100 / 1_000_000, "cached": 0.025 / 1_000_000, "output": 0.400  / 1_000_000},
        "gpt-4.1": {"input": 2.00 / 1_000_000, "cached": 0.50 / 1_000_000, "output": 8.00 / 1_000_000}
    }
    
    if model not in pricing:
        raise ValueError("Model not recognized. Please use one of: GPT-4o, GPT-4o mini, OpenAI o3-mini")
    
    cost = (input_tokens * pricing[model]["input"]) + (output_tokens * pricing[model]["output"]) 
    
    return cost # Formats number to 6 decimal places
