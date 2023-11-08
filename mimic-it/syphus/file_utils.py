"""
file utils
"""

import json
import os
import time

import openai
import random
from litellm import completion

engine = os.environ.get("OPENAI_API_ENGINE", "davinci")


def query_gpt(inputs: dict[str], dataset_name: str) -> tuple[dict[str, str], str]:
    """
    Query the GPT API with the given inputs.
    Returns:
        Response (dict[str, str]): the response from GPT API.
        Input ID (str): the id that specifics the input.
    """
    if dataset_name == "3d.SceneNavigation":
        with open("./candidates.txt") as f:
            candidates = f.readlines()
        cur_candidates = random.sample(candidates, 9)
        cur_candidates_string = "\n".join(cur_candidates)
    messages = [
        {
            "role": "system",
            "content": inputs["system_messages"],
        }
    ]
    # multi-round conversation in the in_context
    messages.extend(inputs["in_context"])
    if dataset_name == "3d.SceneNavigation":
        messages.append(
            {
                "role": "user",
                "content": f"Sentences: {inputs['query_input']['sentences']}\nCandidate activities and the role who want to do these activities:{cur_candidates_string}\nPlease give me three conversations for three activities. Each conversation should have three rounds. You should select activities from the candidates. At the beginning of conversation (after introducing the human role), must giving me the reason why the current activity is selected for this room, in the format:reason:XXX\nPlease ensuring that the assistant should not always answer in the format of listing.",
            },
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": inputs["query_input"]["sentences"],
            },
        )
    succuss = True
    while succuss:
        try:
            response = completion(
                engine=engine,  # defined by os.environ, default engine="chatgpt0301",
                messages=messages,
                temperature=0.7,
                max_tokens=3200,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            succuss = False
        except Exception as e:
            print(f"Error: {e}")
            if "have exceeded call rate limit" in str(e):
                print("Sleeping for 3 seconds")
                succuss = True
                time.sleep(3)
            else:
                succuss = False
                response = {"error_message": str(e)}
    return response, inputs["query_input"]["id"]


def split_question_and_answer(pair_of_answer: str, file_id: str) -> tuple[bool, dict[str, str]]:
    """
    Split the question and answer from the pair of question and answer.
    Args:
        pair_of_answer (str): the pair of question and answer.
        file_id (str): the id of the file.
    """
    try:
        question, answer = pair_of_answer.split("\n")
        question_prefix, question = question.split(": ")
        answer_prefix, answer = answer.split(": ")
        if question_prefix != "Question":
            raise ValueError("The prefix is not Question")
        if answer_prefix != "Answer":
            raise ValueError("The prefix is not Answer")
        return True, {"id": file_id, "question": question, "answer": answer}
    except Exception as e:
        return False, {
            "id": file_id,
            "response": pair_of_answer,
            "error_message": str(e),
        }


def format_output(response: str, file_id: str, dataset_name: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Format the output of ChatGPT.

    Args:
        response (str): the output from ChatGPT.
        file_id (str): the id of the input.

    Returns:
        valid_output (list[dict[str]]): a list of valid output, each item is a dict with keys "id", "question", and "answer".
        invalid_output (list[dict[str]]): a list of invalid output, each item is a dict with keys "id", "response", and "error_message".
    """
    valid_output = []
    invalid_output = []
    if dataset_name == "3d.SceneNavigation":
        for pair_of_answer in response.strip().split("Conversation 1")[1:]:
            is_valid = True
            formatted = {"id": file_id, "results": f"Conversation 1{pair_of_answer}"}
            if is_valid:
                valid_output.append(formatted)
            else:
                invalid_output.append(formatted)
    # elif dataset_name == "video.DenseCaptions":
    else:
        formatted = {"id": file_id, "results": response}
        valid_output.append(formatted)
    # else:
    #     for pair_of_answer in response.strip().split("\n\n"):
    #         is_valid, formatted = split_question_and_answer(pair_of_answer, file_id)
    #         if is_valid:
    #             valid_output.append(formatted)
    #         else:
    #             invalid_output.append(formatted)
    return valid_output, invalid_output


def export_single_output_json(result: dict[str, str], file_name: str, dataset_name: str, duration: float) -> None:
    """
    Export the output of ChatGPT to a json file.

    Args:
    """
    valid_output = []
    invalid_output = []
    output_folder = f"output_{dataset_name}"
    os.makedirs(output_folder, exist_ok=True)
    # if len(data := result["valid_outputs"]) > 0:
    if "valid_outputs" in result:
        valid_output = result["valid_outputs"]
        with open(f"{output_folder}/{file_name}_valid_output.json", "w") as f:
            json.dump(valid_output, f, indent=4)
    if "invalid_outputs" in result:
        invalid_output = result["invalid_outputs"]
        with open(f"{output_folder}/{file_name}_invalid_output.json", "w") as f:
            json.dump(invalid_output, f, indent=4)
    if "error_messages" in result:
        error_messages = result["error_messages"]
        with open(f"{output_folder}/{file_name}_error_messages.json", "w") as f:
            json.dump(error_messages, f, indent=4)
    with open(f"{output_folder}/{file_name}_meta.json", "w") as f:
        meta_data = {}
        meta_data["completion_tokens"] = result["tokens"]["completion_tokens"] if "tokens" in result else 0
        meta_data["prompt_tokens"] = result["tokens"]["prompt_tokens"] if "tokens" in result else 0
        meta_data["total_tokens"] = meta_data["completion_tokens"] + meta_data["prompt_tokens"]
        meta_data["valid_outputs"] = len(valid_output)
        meta_data["invalid_outputs"] = len(invalid_output)
        meta_data["time"] = round(duration, 2)
        json.dump(meta_data, f, indent=4)


def export_output_json(results: list[dict[str, str]], name: str, duration: float) -> None:
    """
    Export the output of ChatGPT to a json file.

    Args:
    """
    valid_output = []
    invalid_output = []
    error_messages = []
    output_folder = f"output_{name}"
    num_completion_tokens = 0
    num_prompt_tokens = 0
    os.makedirs(output_folder, exist_ok=True)
    for result in results:
        if "error_message" in result:
            error_messages.append(result)
            continue
        valid_output.extend(result["valid_outputs"])
        invalid_output.extend(result["invalid_outputs"])
        num_completion_tokens += result["tokens"]["completion_tokens"]
        num_prompt_tokens += result["tokens"]["prompt_tokens"]

    with open(f"{output_folder}/valid_output.json", "w") as f:
        json.dump(valid_output, f, indent=4)
    if len(invalid_output) > 0:
        with open(f"{output_folder}/invalid_output.json", "w") as f:
            json.dump(invalid_output, f, indent=4)

    if len(error_messages) > 0:
        with open(f"{output_folder}/error_messages.json", "w") as f:
            json.dump(error_messages, f, indent=4)
    with open(f"{output_folder}/meta.json", "w") as f:
        json.dump(
            {
                "completion_tokens": num_completion_tokens,
                "prompt_tokens": num_prompt_tokens,
                "total_tokens": num_completion_tokens + num_prompt_tokens,
                "valid_outputs": len(valid_output),
                "invalid_outputs": len(invalid_output),
                "error_messages": len(error_messages),
                "time": round(duration, 2),
                "total_examples": len(results),
            },
            f,
            indent=4,
        )


def save_query_json(inputs: dict[str], name: str) -> None:
    """
    Save the query json to a file.

    Args:
        inputs (dict[str]): the inputs to query the GPT API.
        name (str): the name of the file.
    """
    output_folder = f"output_{name}"
    os.makedirs(output_folder, exist_ok=True)
    with open(f"{output_folder}/query_input.json", "w") as f:
        json.dump(inputs, f, indent=4)
