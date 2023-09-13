import sys

sys.path.append("../..")
from pipeline.evaluation.models.base_model import load_model
from pipeline.evaluation.eval_datasets.base_evel_dataset import load_dataset
import transformers

if __name__ == "__main__":
    evaluator = load_dataset("mhbench", {"dataset_path": "ZhangYuanhan/multi-hop-reasoning"})
    model = load_model("llama_adapter", {"model_path": "/mnt/petrelfs/zhangyuanhan/Otter/pipeline/evaluation/ckpts/7B.pth"})
    evaluator.evaluate(model)

# pip install otter_ai
# other necessary packages
# python -m otter_ai.eval --models=Otter --model_path=luodian/OTTER-Image-MPT --dataset=MultiHopQA
