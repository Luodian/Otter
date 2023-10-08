import base64
import io
from PIL import Image
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os
import numpy as np
from loguru import logger

eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "ocr"],
    "Cognition": ["commonsense", "numerical", "text", "code"]
}

class MMEDataset(object):
    def decode_base64_to_image(self, base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    
    def load_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data
    
    def __init__(self, instruction_file, train_file, image_file, logger_file="output.log"):
        super().__init__()

        logger.add(logger_file)

        self.instruction_file = instruction_file
        self.train_file = train_file
        self.image_file = image_file
        self.instruction_data = self.load_json(self.instruction_file)
        self.train_data = self.load_json(self.train_file)
        self.image_data = self.load_json(self.image_file)
        self.ids = list(self.instruction_data["data"].keys())

        self.category_data = {}
        for idx in range(len(self.ids)):
            id = self.ids[idx]
            category = id.split("_")[0].lower()
            row = self.instruction_data["data"][id]
            question = row["instruction"]
            answer = row["answer"]
            image_id = row["image_ids"][0]
            image = self.decode_base64_to_image(self.image_data[image_id])

            data = {
                "question": question,
                "answer": answer,
                "image": image
            }

            if category in eval_type_dict["Cognition"]:
                eval_type = "Cognition"
            elif category in eval_type_dict["Perception"]:
                eval_type = "Perception"
            else:
                raise ValueError(f"Unknown category {category}")

            if eval_type not in self.category_data:
                self.category_data[eval_type] = {}

            if category not in self.category_data[eval_type]:
                self.category_data[eval_type][category] = {}

            if image_id not in self.category_data[eval_type][category]:
                self.category_data[eval_type][category][image_id] = []
            
            self.category_data[eval_type][category][image_id].append(data)

    def parse_pred_ans(self, pred_ans):
        pred_ans = pred_ans.lower().strip().replace(".", "")
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]
            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"
        return pred_label
    
    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        for key, value in metric_dict.items():
            if isinstance(value, np.int64):
                metric_dict[key] = int(value)

        return metric_dict

    def evaluate(self, model, output_dir='./LaVIN'):
        
        model_score_dict = {}
        for eval_type in self.category_data.keys():
            print("===========", eval_type, "===========")

            scores = 0
            task_score_dict = {}
            for task_name in self.category_data[eval_type].keys():
                img_num = len(self.category_data[eval_type][task_name])
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []
                for image_pair in self.category_data[eval_type][task_name].values():
                    assert len(image_pair) == 2
                    img_correct_num = 0

                    for item in image_pair:
                        question = item["question"]
                        image = item["image"]
                        gt_ans = item["answer"].lower().strip().replace(".", "")
                        pred_ans = self.parse_pred_ans(model.generate(question, image))

                        assert gt_ans in ["yes", "no"]
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)

                        if gt_ans == pred_ans:
                            img_correct_num += 1

                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus

                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                scores += task_score

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, f"{task_name}.json")
                with open(output_path, "w") as f:
                    json.dump(metric_dict, f)

            logger.info(f"total score: {scores}")
            for task_name, score in task_score_dict.items():
                logger.info(f"\t {task_name} score: {score}")
            

        return