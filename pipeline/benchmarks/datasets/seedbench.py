import numpy as np
from tqdm import tqdm
from .base_eval_dataset import BaseEvalDataset
from datasets import load_dataset


class SEEDBenchDataset(BaseEvalDataset):
    def __init__(self, data_path: str = "Otter-AI/SEEDBench", split="test", cache_dir=None):
        super().__init__("SEEDBenchDataset", data_path)
        print("Loading dataset from", data_path)
        self.data = load_dataset(data_path, split=split, cache_dir=cache_dir)

    def evaluate(self, model):
        count = 0
        num_correct = 0
        with tqdm(total=len(self.data), desc="Evaluating") as pbar:
            for data_dict in self.data:
                image = data_dict["image"]
                question = data_dict["question"] + " There are several options:"
                option_index = ["A", "B", "C", "D"]
                for cur_idx in range(4):
                    question += f" {option_index[cur_idx]}. {data_dict[f'choice_{option_index[cur_idx].lower()}']}"

                answer = data_dict["answer"]
                options = [
                    data_dict["choice_a"],
                    data_dict["choice_b"],
                    data_dict["choice_c"],
                    data_dict["choice_d"],
                ]

                option_losses = []
                for idx, option in enumerate(options):
                    option = option_index[idx] + ". " + option
                    loss = model.eval_forward(question, option, image)
                    option_losses.append(loss.item())

                prediction_idx = np.argmin(option_losses)
                prediction = ["A", "B", "C", "D"][prediction_idx]
                if prediction == answer:
                    num_correct += 1
                count += 1

                accuracy = num_correct / count * 100
                pbar.set_postfix(accuracy=f"{accuracy:.2f}")
                pbar.update(1)

        accuracy = num_correct / count * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy
