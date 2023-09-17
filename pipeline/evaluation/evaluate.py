import sys
import argparse
import os
import yaml

sys.path.append("../..")
from pipeline.evaluation.models.base_model import load_model
from pipeline.evaluation.eval_datasets.base_evel_dataset import load_dataset
import transformers

def get_info(info):
    if "name" not in info:
        raise ValueError("Model name is not specified.")
    name = info["name"]
    info.pop("name")
    return name, info

def load_models(model_infos):
    for model_info in model_infos:
        name, info = get_info(model_info)
        model = load_model(name, info)
        yield model

def load_datasets(dataset_infos):
    for dataset_info in dataset_infos:
        name, info = get_info(dataset_info)
        dataset = load_dataset(name, info)
        yield dataset
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", type=str, help="Path to the config file.")
    phrased_args = args.parse_args()
    with open(phrased_args.config, "r") as f:
        config = yaml.safe_load(f)
    
    for model in load_models(config["models"]):
        print("------------------------------------------------------------")
        print(f"Evaluating model {model.name}")
        for dataset in load_datasets(config["datasets"]):
            print(f"Evaluating with dataset {dataset.name}")
            dataset.evaluate(model)
            print(f"Finished evaluating with dataset {dataset.name}")
        print(f"Finished evaluating model {model.name}")
        print("------------------------------------------------------------")


# pip install otter_ai
# other necessary packages
# python -m otter_ai.eval --models=Otter --model_path=luodian/OTTER-Image-MPT --dataset=MultiHopQA
