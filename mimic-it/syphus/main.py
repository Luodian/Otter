import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union

import openai
from tqdm import tqdm

from abstract_dataset import get_dataset_by_path
from file_utils import (
    save_query_json,
    export_output_json,
    format_output,
    query_gpt,
)


def task(inputs: Dict[str, Union[str, Dict[str, Union[str, int]]]]) -> Dict[str, Union[Dict[str, int], List[str]]]:
    global dataset_name
    try:
        gpt_output, file_id = query_gpt(inputs, dataset_name)
        tokens = dict(gpt_output["usage"])
        valid_outputs, invalid_outputs = format_output(gpt_output["choices"][0]["message"]["content"], file_id, dataset_name)
        result = {
            "tokens": tokens,
            "valid_outputs": valid_outputs,
            "invalid_outputs": invalid_outputs,
        }
    except Exception as e:
        result = {"error_message": str(e)}
    return result


if __name__ == "__main__":
    openai.api_type = os.environ.get("OPENAI_API_TYPE", "local")
    openai.api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000")
    openai.api_version = os.environ.get("OPENAI_API_VERSION", "2020-04-01")
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Path to the dataset class.")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads.")
    parser.add_argument("--slice_start", type=int, default=0, help="Dryrun test, on querying N samples.")
    parser.add_argument("--slice_end", type=int, default=-1, help="Dryrun test, on querying N samples.")
    parser.add_argument("--random_sample", action="store_true", help="Random sample.")
    parser.add_argument("--dataset_version", default="v1", help="Dataset version.")
    parser.add_argument("--prompt_path", help="Path to the prompt file.")
    parser.add_argument("--query_inputs_path", "-in", help="Path to the query input file.")

    args = parser.parse_args()
    dataset_args = {}
    if args.prompt_path is not None:
        dataset_args["prompt_path"] = args.prompt_path
    if args.query_inputs_path is not None:
        dataset_args["query_inputs_path"] = args.query_inputs_path
    dataset = get_dataset_by_path(args.name, dataset_args)
    dataset_name = args.name
    dataset = list(dataset)
    if args.random_sample:
        import random

        random.shuffle(dataset)
    if args.slice_end > 0:
        dataset = dataset[args.slice_start : args.slice_end]
    results = []
    query_inputs = []
    start_time = time.time()

    if args.num_threads == 0:
        progress_bar = tqdm(total=len(dataset), unit="task")
        for n, d in enumerate(dataset):
            query_inputs.append(d["query_input"])
            results.append(task(d))
            progress_bar.update(1)
        progress_bar.close()
    else:
        progress_bar = tqdm(total=len(dataset))

        def update_progress(_):
            progress_bar.update(1)

        # Submit the tasks to the thread pool
        progress_bar = tqdm(total=len(dataset), unit="task")
        batch_size = args.num_threads
        for i in range(0, len(dataset), batch_size):
            # Create a thread pool with the specified number of threads
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                current_batch = dataset[i : i + batch_size]
                futures = [executor.submit(task, d) for d in current_batch]
                query_inputs.extend([d["query_input"] for d in current_batch])
                # Retrieve the results as they become available
                for future, num in zip(futures, dataset):
                    results.append(future.result())
                    progress_bar.update(1)
        progress_bar.close()

    duration = time.time() - start_time
    save_query_json(query_inputs, f"{dataset_name}_{args.dataset_version}")
    export_output_json(results, f"{dataset_name}_{args.dataset_version}", duration)
    print(f"Total time: {duration:.2f}s")
