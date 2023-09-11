import sys
sys.path.append("../..")
from pipeline.evaluation.datasets.mmbench import MMBenchDataset
from pipeline.evaluation.models.model import load_model
import transformers

if __name__ == "__main__":
    model_info = {
        "model_path": "/home/luodian/projects/checkpoints/idefics-9b-instruct",
    }
    evaluator = MMBenchDataset("/home/luodian/azure_storage/otter/mimicit/MMBench/mmbench_test_20230712.tsv")
    model = load_model("idefics", model_info)
    evaluator.evaluate(model)

# pip install otter_ai
# other necessary packages
# python -m otter_ai.eval --models=Otter --model_path=luodian/OTTER-Image-MPT --dataset=MultiHopQA
