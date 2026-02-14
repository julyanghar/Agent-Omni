"""
Agent-Omni Adapter for Daily-Omni Benchmark

This module provides an adapter function that bridges the Daily-Omni evaluation
framework with Agent-Omni's graph-based reasoning system.
"""

import sys
import os
import re
import traceback

# Add Agent-Omni root directory to path
_agent_omni_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, _agent_omni_root)

# Set config path if not already set
if 'CONFIG_PATH' not in os.environ:
    _config_path = os.path.join(_agent_omni_root, 'test', 'config.yaml')
    if os.path.exists(_config_path):
        os.environ['CONFIG_PATH'] = _config_path
    else:
        _config_path = os.path.join(_agent_omni_root, 'src', 'config.yaml')
        if os.path.exists(_config_path):
            os.environ['CONFIG_PATH'] = _config_path

from src.state import State
from src.graph import _graph
from src.config import config


def extract_answer_letter(text):
    """
    Extract the first occurrence of A, B, C, or D from the text.
    
    Args:
        text: The answer text from Agent-Omni
        
    Returns:
        str: The extracted letter (A, B, C, or D), or error string if not found
    """
    if not text or not isinstance(text, str):
        return "error_no_answer"
    
    # Try to find a capital letter A, B, C, or D
    # First, try to find standalone letters or letters at the start
    match = re.search(r'\b([A-D])\b', text.upper())
    if match:
        return match.group(1)
    
    # If not found, try to find any A-D at the beginning of the string
    match = re.match(r'\s*([A-D])', text.upper())
    if match:
        return match.group(1)
    
    # If still not found, return error
    return "error_answer_extraction"


def ask_agent_omni(question, choices, video_path):
    """
    Adapter function to call Agent-Omni for Daily-Omni benchmark evaluation.
    
    This function:
    1. Formats the question and choices into a query for Agent-Omni
    2. Creates a State object with the video and query
    3. Calls Agent-Omni's graph to perform reasoning
    4. Extracts the answer letter (A/B/C/D) from the final answer
    
    Args:
        question (str): The question text
        choices (str): The multiple choice options (e.g., "A) Option 1, B) Option 2, ...")
        video_path (str): Path to the video file
        
    Returns:
        str: The answer letter (A, B, C, or D), or an error string starting with "error_"
    """
    try:
        print(f"Processing (Agent-Omni): {os.path.basename(video_path) if video_path else 'No video'}")
        
        # Check if video exists
        if video_path and not os.path.exists(video_path):
            print(f"Warning: Video file not found at {video_path}")
            return "error_video_not_found"
        
        # Format the query with explicit instruction to return only a letter
        query = f"""Your task is to accurately answer multiple-choice questions based on the given video.
Select the single most accurate answer from the given choices.

Question: {question}
Choices: {choices}

Your answer should be ONLY a capital letter representing your choice: A, B, C, or D. Don't generate any other text."""
        
        # Create State object
        # Get max_round_num from config, default to 3
        max_rounds = 3
        if "system" in config and "max_rounds" in config["system"]:
            max_rounds = config["system"]["max_rounds"]
        elif "system" in config and "retry_times" in config["system"]:
            # Fallback: use retry_times if max_rounds not available
            max_rounds = min(config["system"]["retry_times"], 5)  # Cap at 5
        
        state = State(messages=[], max_round_num=max_rounds)
        state["query"] = query
        
        # Set video path (as a list, as expected by Agent-Omni)
        if video_path:
            state["video"] = [video_path]
        else:
            state["video"] = []
        
        # Set other required fields
        state["text"] = []
        state["audio"] = []
        state["image"] = []
        
        # Call Agent-Omni graph
        try:
            end_state = _graph.batch([state])
            
            # Extract final answer
            if not end_state or len(end_state) == 0:
                print("Error: Agent-Omni returned empty state")
                return "error_empty_state"
            
            final_state = end_state[0]
            
            # Check if decision_result exists
            if "decision_result" not in final_state or final_state["decision_result"] is None:
                print("Error: No decision_result in Agent-Omni output")
                return "error_no_decision_result"
            
            decision_result = final_state["decision_result"]
            
            # Extract final_answer
            if "final_answer" not in decision_result or decision_result["final_answer"] is None:
                print("Error: No final_answer in decision_result")
                return "error_no_final_answer"
            
            final_answer = decision_result["final_answer"]
            
            # Extract answer letter
            answer = extract_answer_letter(final_answer)
            
            if answer.startswith("error_"):
                print(f"Warning: Could not extract answer letter from: {final_answer}")
            
            return answer
            
        except Exception as e:
            print(f"Error calling Agent-Omni graph: {e}")
            traceback.print_exc()
            return "error_graph_execution"
            
    except Exception as e:
        print(f"Error in ask_agent_omni: {e}")
        traceback.print_exc()
        return "error_adapter_exception"

