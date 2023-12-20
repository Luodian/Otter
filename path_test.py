import orjson
import os
from tqdm import tqdm

with open("/mnt/petrelfs/share_data/zhangyuanhan/m3it/LA_MIX/LA_MIX_COCO.json") as f:
    cache_train_config = orjson.loads(f.read())
for _ in tqdm(cache_train_config["data"]):
    # import pdb;pdb.set_trace() 
    cur_image_id = cache_train_config["data"][_]["image_ids"][0]
    real_cur_image_id = cur_image_id.split("_")[-1]
    if "GQA_" in cur_image_id:
        cur_image_path = f"/mnt/petrelfs/share_data/basemodel/dataset/multimodality/gqa/images/{real_cur_image_id}.jpg"
    elif "COCO" in cur_image_id:
        cur_image_path = f"/mnt/petrelfs/share_data/basemodel/dataset/multimodality/coco/train2017/{real_cur_image_id}.jpg"
    elif "OCR_VQA" in cur_image_id:
        cur_image_path = f"/mnt/petrelfs/zhangyuanhan/LLaVA/playground/data/ocr_vqa/images/{real_cur_image_id}.jpg"   
    elif "TEXTVQA" in cur_image_id:   
        cur_image_path = f"/mnt/petrelfs/share_data/basemodel/dataset/multimodality/textqa/train_images/{real_cur_image_id}.jpg"                     
    elif "VG" in cur_image_id:
        if "VG_100K_2_" in cur_image_id:
            real_cur_image_id = cur_image_id.replace("VG_IMG_","").replace("VG_100K_2_","VG_100K_2/")
        else:
            real_cur_image_id = cur_image_id.replace("VG_IMG_","").replace("VG_100K_","VG_100K/")       
        cur_image_path = f"/mnt/petrelfs/zhangyuanhan/LLaVA/playground/data/vg/{real_cur_image_id}.jpg"

    if not os.path.exists(cur_image_path):
        import pdb;pdb.set_trace()


# import pdb;pdb.set_trace()