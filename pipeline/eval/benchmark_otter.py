import requests
import torch
import transformers
import json
from PIL import Image
from otter.modeling_otter import OtterForConditionalGeneration
import argparse
from tqdm import tqdm

requests.packages.urllib3.disable_warnings()


def get_image(url: str) -> Image.Image:
    """
    Get image from url

    Args:
        url (str): url of the image

    Returns:
        Image.Image: PIL Image
    """
    return Image.open(requests.get(url, stream=True, verify=False).raw)


def get_formatted_prompt(prompt: str) -> str:
    """
    Format prompt for GPT

    Args:
        prompt (str): prompt to be formatted

    Returns:
        str: formatted prompt
    """
    return f"<image> User: {prompt} GPT:<answer>"


def get_response(url: str, prompt: str, model=None, image_processor=None) -> str:
    """
    Get the response of single image and prompt from the model

    Args:
        url (str): url of the image
        prompt (str): the prompt (no need to be formatted)

    Returns:
        str: response of the model
    """
    query_image = get_image(url)
    vision_x = (
        image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"]
        .unsqueeze(1)
        .unsqueeze(0)
    )
    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x["input_ids"].to(model.device),
        attention_mask=lang_x["attention_mask"].to(model.device),
        max_new_tokens=256,
        num_beams=3,
        no_repeat_ngram_size=3,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output


def generate_html(output_file, model_version_or_tag):
    import json

    # Load the data from the JSON file
    with open(output_file, "r") as f:
        data = json.load(f)

    # Start the HTML file
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmarking various ver. of Otter</title>
        <style>
            .column {{
                float: left;
                width: 33.33%;
                padding: 5px;
                box-sizing: border-box;
            }}
            .row::after {{
                content: "";
                clear: both;
                display: table;
            }}
            img {{
                width: 338px;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <h1>{}</h1>
    """

    html = html.format(model_version_or_tag)

    # Add headers
    html += """
        <div class="row">
            <div class="column">
                <h2>Image</h2>
            </div>
            <div class="column">
                <h2>Instruction</h2>
            </div>
            <div class="column">
                <h2>Response</h2>
            </div>
        </div>
    """

    # Add the data to the HTML
    for item in data:
        html += """
        <div class="row">
            <div class="column">
                <img src="{image}" alt="Image">
            </div>
            <div class="column">
                {instruction}
            </div>
            <div class="column">
                {response}
            </div>
        </div>
        """.format(
            **item
        )

    # Close the HTML tags
    html += """
    </body>
    </html>
    """

    # Write the HTML string to a file
    output_html_path = output_file.replace(".json", ".html")
    with open(output_html_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path_or_name",
        type=str,
        default="luodian/otter-9b-hf",
        help="Path or name of the model (HF format)",
    )
    parser.add_argument(
        "--model_version_or_tag",
        type=str,
        default="apr25_otter",
        help="Version or tag of the model",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="evaluation/sample_questions.json",
        help="Path of the input file",
    )
    args = parser.parse_args()

    model = OtterForConditionalGeneration.from_pretrained(
        args.model_path_or_name, device_map="auto"
    )
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()

    responses = []
    with open(args.input_file) as f:
        data = json.load(f)
        progress_bar = tqdm(total=len(data["input"]))
        for item in data["input"]:
            print("=" * 50)
            print(f"Processing {item['image']} with prompt {item['instruction']}")
            response = get_response(
                item["image"], item["instruction"], model, image_processor
            )
            print(f"Response: {response}")
            responses.append(
                {
                    "image": item["image"],
                    "instruction": item["instruction"],
                    "response": response,
                }
            )
            progress_bar.update(1)
    json.dump(
        responses,
        open(f"./evaluation/{args.model_version_or_tag}_outputs.json", "w"),
        indent=4,
    )

    generate_html(
        f"./evaluation/{args.model_version_or_tag}_outputs.json",
        args.model_version_or_tag,
    )
