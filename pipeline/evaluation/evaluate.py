import sys

sys.path.append("../..")
from pipeline.evaluation.evaluator.mmbench import MMBenchEvaluator

if __name__ == "__main__":
    model_info = {
        "model_path": "/data/pufanyi/training_data/checkpoints/idefics-9b-instruct",
    }
    evaluator = MMBenchEvaluator("/data/pufanyi/training_data/MMBench/mmbench_test_20230712.tsv")
    model = load_model("idefics", model_info)
    evaluator.evaluate(model)

# pip install otter_ai
# other necessary packages
# python -m otter_ai.eval --models=Otter --model_path=luodian/OTTER-Image-MPT --dataset=MMBench