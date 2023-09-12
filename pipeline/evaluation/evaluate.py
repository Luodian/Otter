import sys

sys.path.append("../..")
from pipeline.evaluation.eval_datasets.mmbench import MMBenchDataset
from pipeline.evaluation.eval_datasets.mhbench import MHBenchDataset
from pipeline.evaluation.models.model import load_model
import transformers

if __name__ == "__main__":
    model_info = {
        "model_path": "/mnt/petrelfs/zhangyuanhan/Otter/checkpoint/idefics-9b-instruct",
    }
    evaluator = MHBenchDataset("ZhangYuanhan/multi-hop-reasoning")
    # model = load_model("idefics", model_info)
    model = load_model("llama_adapter",{})
    import pdb;pdb.set_trace()
    evaluator.evaluate(model)

# pip install otter_ai
# other necessary packages
# python -m otter_ai.eval --models=Otter --model_path=luodian/OTTER-Image-MPT --dataset=MultiHopQA
