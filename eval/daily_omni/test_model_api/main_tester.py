# main_tester.py

import os
import json
import time
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from functools import partial # For passing fixed args to worker
from tqdm import tqdm

from . import test_config as config
from . import test_utils as utils


os.environ['GEMINI_API_KEY_DEFAULT'] = 'AIzaSyBPGiYIUgIy-mvAF2ifVNnileWksEtNE4k'


# Mapping from model_type string to the corresponding function in utils
MODEL_FUNCTION_MAP = {
    'gemini_av': utils.ask_gemini_av,
    'gemini_visual': utils.ask_gemini_visual,
    'gpt4o_visual': utils.ask_gpt4o_visual,
    'gpt4o_text': utils.ask_gpt4o_text,
    'deepseek_text': utils.ask_deepseek_text,
    'agent_omni': utils.ask_agent_omni,
}

def process_single_item_worker(item_data, model_function, video_base_dir=None):
    """
    Worker function for parallel processing. Calls the appropriate model function.
    Args:
        item_data (dict): The QA item dictionary.
        model_function (callable): The function from utils_tester to call (e.g., utils.ask_gemini_av).
        video_base_dir (str, optional): Base directory for video files. If None, uses config.BASE_VIDEO_DIR.
    Returns:
        dict: Result dictionary including answer, correctness, status flags.
    """
    question = item_data.get('Question')
    choices = item_data.get('Choice')
    correct_answer_char = item_data.get('Answer')
    video_id = item_data.get('video_id')
    qa_type = item_data.get('Type')
    video_category = item_data.get('video_category')
    video_duration = item_data.get('video_duration')

    # Basic validation of item data
    if not all([question, choices, correct_answer_char, video_id, qa_type, video_category, video_duration]):
        # print(f"Warning: Skipping item due to missing fields. Item ID: {video_id if video_id else 'Unknown'}")
        return {
            "skipped": True, "video_id": video_id, "reason": "Missing fields",
            "qa_type": qa_type, "video_category": video_category, "video_duration": video_duration
        }

    video_path = utils.get_video_path(video_id, base_path=video_base_dir)

    # Check video existence *before* calling API, except for text-only models
    if model_function not in [utils.ask_gpt4o_text, utils.ask_deepseek_text]:
        if not os.path.exists(video_path):
            # print(f"Warning: Video file not found for ID {video_id} at {video_path}")
            return {
                "skipped": True, "video_id": video_id, "reason": "Video file not found",
                "question": question, "api_answer": "error_video_not_found", "correct_answer": correct_answer_char,
                "is_correct": False, "api_call_failed": True, # Treat missing video as a failure
                "qa_type": qa_type, "video_category": video_category, "video_duration": video_duration
            }

    # Call the specific model's API function
    api_answer = model_function(question, choices, video_path)

    # Evaluate result
    api_call_failed = isinstance(api_answer, str) and api_answer.startswith("error_")
    is_correct = False
    if not api_call_failed:
        is_correct = utils.evaluate_answer(api_answer, correct_answer_char)

    return {
        "skipped": False,
        "video_id": video_id,
        "question": question,
        "api_answer": api_answer,
        "correct_answer": correct_answer_char,
        "is_correct": is_correct,
        "api_call_failed": api_call_failed,
        "qa_type": qa_type,
        "video_category": video_category,
        "video_duration": video_duration
    }


def run_tests(model_type, execution_mode, qa_json_path, max_items=None, video_base_dir=None, output_dir=None):
    """
    Main function to orchestrate loading data and running tests.
    
    Args:
        model_type (str): Type of model to test.
        execution_mode (str): Execution mode ('sequential' or 'parallel').
        qa_json_path (str): Path to QA JSON file.
        max_items (int, optional): Maximum number of items to process.
        video_base_dir (str, optional): Base directory for video files. If None, uses config.BASE_VIDEO_DIR.
        output_dir (str, optional): Output directory for results. If None, uses default: eval/daily_omni/eval_results
    """
    # Use provided video_base_dir or fall back to config
    if video_base_dir is None:
        video_base_dir = config.BASE_VIDEO_DIR
    # Convert to absolute path
    video_base_dir = os.path.abspath(video_base_dir)
    
    print(f"--- Starting Tests ---")
    print(f"Model Type: {model_type}")
    print(f"Execution Mode: {execution_mode}")
    print(f"QA Data Path: {qa_json_path}")
    print(f"Video Base Path: {video_base_dir}")

    if model_type not in MODEL_FUNCTION_MAP:
        print(f"Error: Invalid model_type '{model_type}'. Valid options are: {list(MODEL_FUNCTION_MAP.keys())}")
        return

    model_function_to_call = MODEL_FUNCTION_MAP[model_type]

    # Load data
    all_data = utils.load_json_data(qa_json_path)
    if not all_data:
        print("Error: No data loaded. Exiting.")
        return

    # Optional: Limit number of items for testing
    if max_items is not None and max_items > 0:
        print(f"Limiting processing to first {max_items} items.")
        data_to_process = all_data[:max_items]
    else:
        data_to_process = all_data

    total_items_requested = len(data_to_process)
    print(f"Found {total_items_requested} items to process.")
    if total_items_requested == 0:
        print("No items selected for processing.")
        return

    all_results = []
    start_time = time.time()

    if execution_mode == 'sequential':
        print("Running in Sequential mode...")
        for item in tqdm(data_to_process, desc="Processing Sequentially"):
            result = process_single_item_worker(item, model_function_to_call, video_base_dir=video_base_dir)
            all_results.append(result)
            # Optional: print intermediate result details
            if not result.get("skipped"):
                 print(f"\n  Q: {result['question'][:50]}... -> A: {result['api_answer']} (Correct: {result['correct_answer']}, Status: {'OK' if not result['api_call_failed'] else 'FAIL'})")
            else:
                 print(f"\n  Skipped: {result.get('reason')}")


    elif execution_mode == 'parallel':
        print("Running in Parallel mode...")
        num_workers = config.MAX_WORKERS if config.MAX_WORKERS else min(os.cpu_count(), total_items_requested)
        print(f"Using {num_workers} workers.")

        # Use partial to fix the model_function and video_base_dir arguments for the worker
        worker_func_with_model = partial(process_single_item_worker, model_function=model_function_to_call, video_base_dir=video_base_dir)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks
            futures = [executor.submit(worker_func_with_model, item) for item in data_to_process]

            # Process results as they complete with tqdm progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Parallel"):
                try:
                    result = future.result()
                    all_results.append(result)
                     # Optional: print intermediate result details
                    if not result.get("skipped"):
                         print(f"\n  Q: {result['question'][:50]}... -> A: {result['api_answer']} (Correct: {result['correct_answer']}, Status: {'OK' if not result['api_call_failed'] else 'FAIL'})")
                    else:
                         print(f"\n  Skipped: {result.get('reason')}")
                except Exception as e:
                    print(f"\nError retrieving result from worker process: {e}")
                    # Append a generic failure result if needed
                    all_results.append({"skipped": True, "reason": f"Worker error: {e}", "api_call_failed": True})
    else:
        print(f"Error: Invalid execution_mode '{execution_mode}'. Use 'sequential' or 'parallel'.")
        return

    end_time = time.time()
    print(f"\n--- Testing Complete ---")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    # Calculate and print statistics
    stats_dict = utils.print_statistics(all_results, total_items_requested)
    
    # Save results to files
    _save_evaluation_results(
        all_results=all_results,
        stats_dict=stats_dict,
        model_type=model_type,
        execution_mode=execution_mode,
        qa_json_path=qa_json_path,
        total_items_requested=total_items_requested,
        processing_time=end_time - start_time,
        output_dir=output_dir
    )


def _save_evaluation_results(all_results, stats_dict, model_type, execution_mode, qa_json_path, total_items_requested, processing_time, output_dir=None):
    """
    Save evaluation results to two files:
    1. Detailed results file: Each question with video_id, question, answer, ground_truth
    2. Statistics file: Overall statistics and breakdowns
    
    Args:
        all_results: List of evaluation results
        stats_dict: Statistics dictionary
        model_type: Model type
        execution_mode: Execution mode
        qa_json_path: Path to QA JSON file
        total_items_requested: Total items requested
        processing_time: Processing time in seconds
        output_dir: Output directory for results. If None, uses default: eval/daily_omni/eval_results
    """
    # Generate output directory and filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Default output directory: eval/daily_omni/eval_results (relative to Agent-Omni root)
    if output_dir is None:
        # Get the directory of this file (test_model_api/main_tester.py)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to daily_omni directory, then to eval_results
        output_dir = os.path.join(os.path.dirname(current_file_dir), "eval_results")
    
    # Convert to absolute path
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # File 1: Detailed results (JSON)
    detailed_filename = f"detailed_results_{model_type}_{timestamp}.json"
    detailed_path = os.path.join(output_dir, detailed_filename)
    
    # File 2: Statistics (JSON)
    stats_filename = f"statistics_{model_type}_{timestamp}.json"
    stats_path = os.path.join(output_dir, stats_filename)
    
    # Prepare detailed results
    detailed_results = []
    for result in all_results:
        detailed_item = {
            "video_id": result.get("video_id", "unknown"),
            "question": result.get("question", ""),
            "model_answer": result.get("api_answer", ""),
            "ground_truth": result.get("correct_answer", ""),
            "is_correct": result.get("is_correct", False),
            "qa_type": result.get("qa_type", ""),
            "video_category": result.get("video_category", ""),
            "video_duration": result.get("video_duration", ""),
            "skipped": result.get("skipped", False),
            "skip_reason": result.get("reason", ""),
            "api_call_failed": result.get("api_call_failed", False)
        }
        detailed_results.append(detailed_item)
    
    # Save detailed results
    try:
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Detailed results saved to: {detailed_path}")
    except Exception as e:
        print(f"\n✗ Error saving detailed results: {e}")
    
    # Prepare statistics
    statistics = {
        "model_type": model_type,
        "execution_mode": execution_mode,
        "timestamp": timestamp,
        "processing_time_seconds": round(processing_time, 2),
        "overall": {
            "items_requested": total_items_requested,
            "items_skipped": stats_dict.get("skipped", 0),
            "items_attempted": stats_dict.get("total_processed", 0),
            "api_call_failures": stats_dict.get("api_failures", 0),
            "items_evaluated": stats_dict.get("items_for_accuracy", 0),
            "overall_correct": stats_dict.get("overall_correct", 0),
            "overall_accuracy": stats_dict.get("overall_accuracy", 0.0)
        },
        "accuracy_by_qa_type": stats_dict.get("accuracy_by_type", {}),
        "accuracy_by_video_category": stats_dict.get("accuracy_by_category", {}),
        "accuracy_by_video_duration": stats_dict.get("accuracy_by_duration", {})
    }
    
    # Save statistics
    try:
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"✓ Statistics saved to: {stats_path}")
    except Exception as e:
        print(f"✗ Error saving statistics: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA tests against various video/text models.")
    parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL_TYPE,
        choices=list(MODEL_FUNCTION_MAP.keys()),
        help="Type of model to test."
    )
    parser.add_argument(
        "--mode", type=str, default=config.DEFAULT_EXECUTION_MODE,
        choices=['sequential', 'parallel'],
        help="Execution mode."
    )
    parser.add_argument(
        "--qa_file", type=str, default=config.DEFAULT_QA_JSON_PATH,
        help="Path to the QA JSON file."
    )
    parser.add_argument(
        "--max_items", type=int, default=None,
        help="Maximum number of QA items to process (for testing)."
    )

    args = parser.parse_args()

    # --- Environment Variable Checks (Optional but Recommended) ---
    print("Checking for necessary API keys (set via environment variables or in config_tester.py)...")
    required_keys = {
        'gemini_av': ['GEMINI_API_KEY'],
        'gemini_visual': ['GEMINI_API_KEY'],
        'gpt4o_visual': ['GPT4O_API_KEY', 'GPT4O_BASE_URL'],
        'gpt4o_text': ['GPT4O_API_KEY', 'GPT4O_BASE_URL'],
        'deepseek_text': ['DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL']
    }
    missing_configs = []
    for req_key in required_keys.get(args.model, []):
         # Check if the key exists as a non-empty attribute in the config module
         if not hasattr(config, req_key) or not getattr(config, req_key):
             missing_configs.append(req_key)

    if missing_configs:
         print(f"ERROR: Missing configuration(s) for model '{args.model}': {', '.join(missing_configs)}")
         print("Please set them as environment variables or directly in config_tester.py.")
         exit(1)
    else:
         print("API key configuration check passed.")
         # Check for external dependencies if needed
         if args.model == 'gemini_visual':
              print(f"Note: '{args.model}' requires ffmpeg to be installed and accessible at '{config.FFMPEG_PATH}'.")
         if args.model == 'gpt4o_visual':
              print(f"Note: '{args.model}' requires opencv-python (cv2) to be installed.")


    # Run the tests
    run_tests(
        model_type=args.model,
        execution_mode=args.mode,
        qa_json_path=args.qa_file,
        max_items=args.max_items
    )

