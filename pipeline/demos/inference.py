import argparse
import json
import sys
import datetime
import requests
import yaml

from .demo_models import TestIdefics, TestOtter, TestOtterIdefics
from .demo_utils import get_image, print_colored

requests.packages.urllib3.disable_warnings()

import pytz

# Initialize the time zone
utc_plus_8 = pytz.timezone('Asia/Singapore')  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
# Get the current time in UTC
utc_now = pytz.utc.localize(datetime.utcnow())
# Convert to UTC+8
utc_plus_8_time = utc_now.astimezone(utc_plus_8)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="otter", required=True, help="The model name.")
    parser.add_argument("--checkpoint", type=str, help="The path to the checkpoint.")
    parser.add_argument("--output_dir", type=str, help="The dir path to the output file.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model_name == "otter":
        model = TestOtter(checkpoint=args.checkpoint)
    elif args.model_name == "otter_idefics":
        model = TestOtterIdefics(checkpoint=args.checkpoint)
    elif args.model_name == "idefics":
        model = TestIdefics(checkpoint=args.checkpoint)

    while True:
        yaml_file = input("Enter the path to the yaml file: (or 'q' to quit): ")
        if yaml_file == "q":
            break
        with open(yaml_file, "r") as file:
            test_data_list = yaml.safe_load(file)

        cur_date = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        log_json_path = f"{args.output_dir}/inference_log_{cur_date}.json"
        log_json = {
            "model_name": args.model_name,
            "checkpoint": args.checkpoint,
            "results": {},
        }
        for test_id, test_data in enumerate(test_data_list):
            image_path = test_data.get("image_path", "")
            question = test_data.get("question", "")

            image = get_image(image_path)
            no_image_flag = not bool(image_path)

            response = model.generate(prompt=question, image=image, no_image_flag=no_image_flag)

            # Print results to console
            print(f"image_path: {image_path}")
            print_colored(f"question: {question}", color_code="\033[92m")
            print_colored(f"answer: {response}", color_code="\033[94m")
            print("-" * 150)

            log_json['results'].update(
                {
                    str(test_id).zfill(3): {
                        "image_path": image_path,
                        "question": question,
                        "answer": response,
                    }
                }
            )

        with open(log_json_path, "w") as file:
            json.dump(log_json, file, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()
