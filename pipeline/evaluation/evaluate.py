import sys

sys.path.append("../..")


from pipeline.evaluation.evaluator.mmbench import MMBenchEvaluator
from pipeline.evaluation.models.idefics import Idefics


if __name__ == "__main__":
    # model = Otter("/data/pufanyi/training_data/checkpoints/OTTER-Image-MPT7B")
    evaluator = MMBenchEvaluator("/data/pufanyi/training_data/MMBench/mmbench_test_20230712.tsv")
    model = Idefics("/data/pufanyi/training_data/checkpoints/idefics-9b-instruct")
    evaluator.evaluate(model)

# pip install otter_ai
# other necessary packages
# python -m otter_ai.eval --models=Otter --model_path=luodian/OTTER-Image-MPT --dataset=MMBench
