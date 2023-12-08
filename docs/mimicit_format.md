# Breaking Down the MIMIC-IT Format

❗❗❗We changed previous `images.json` to `images.parquet`. They are all containing multiple `key:base64` pairs but the later one would consume far less CPU memory and faster during loading with `pandas.Dataframe`. It enables us to train with larger datasets more conviently.

We mainly use one integrate dataset format and we refer it to MIMIC-IT format since. You can convert any of your datasets into the same format like the following mentioned (two files for each dataset).

We use the following data yaml file to indicate the data group and dataset we used in training. Within this data yaml file, for each dataset, you could assign the path of the instruction json file and the image parquet file, and also the number of samples you want to use. The number of samples within each group will be uniformly sampled, and the `number_samples / total_numbers`` will decide sampling ratio of each dataset.

```yaml
IMAGE_TEXT: # Group name should be in [IMAGE_TEXT, TEXT_ONLY, IMAGE_TEXT_IN_CONTEXT]
  LADD: # Dataset name can be assigned at any name you want
    mimicit_path: azure_storage/json/LA/LADD_instructions.json # Path of the instruction json file
    images_path: azure_storage/Parquets/LA.parquet # Path of the image parquet file
    num_samples: -1 # Number of samples you want to use, -1 means use all samples, if not set, default is -1.
  LACR_T2T:
    mimicit_path: azure_storage/json/LA/LACR_T2T_instructions.json
    images_path: azure_storage/Parquets/LA.parquet
    num_samples: -1
  M3IT_CAPTIONING:
    mimicit_path: azure_storage/json/M3IT/captioning/coco/coco_instructions.json
    images_path: azure_storage/Parquets/coco.parquet
    num_samples: 20000

TEXT_ONLY:
  LIMA:
    mimicit_path: azure_storage/json/LANG_Only/LIMA/LIMA_instructions_max_1K_tokens.json
    num_samples: 20000
  SHAREGPT:
    mimicit_path: azure_storage/json/LANG_Only/SHAREGPT/SHAREGPT_instructions_max_1K_tokens.json
    num_samples: 10000
  AL:
    mimicit_path: azure_storage/json/LANG_Only/AL/AL_instructions_max_1K_tokens.json
    num_samples: 20000
```

The data yaml file mainly include two groups of data (1) IMAGE_TEXT (2) TEXT_ONLY. 

For each group, one dataset contains the `instruction.json` file and `images.parquet` file. You can browse the `instruction.json` file at [here](https://entuedu-my.sharepoint.com/:f:/g/personal/libo0013_e_ntu_edu_sg/Eo9bgNV5cjtEswfA-HfjNNABiKsjDzSWAl5QYAlRZPiuZA?e=nNUhJH) and the `images.parquet` file at [here](https://entuedu-my.sharepoint.com/:f:/g/personal/libo0013_e_ntu_edu_sg/EmwHqgRtYtBNryTcFmrGWCgBjvWQMo1XeCN250WuM2_51Q?e=sCymXx). We will provide more at the same Onedrive folder gradually due to the limited internet bandwith, you send emails to push us.

You are also welcome to make your own data into this format, let's breakdown what's inside them:

## DallE3_instructions.json
```
{
	"meta": { "version": "0.0.1", "time": "2023-10-29", "author": "Jingkang Yang" },
	"data": {
		"D3_INS_000000": {
			"instruction": "What do you think is the prompt for this AI-generated picture?",
			"answer": "photo of a gigantic hand coming from the sky reaching out people who are holding hands at a beach, there is also a giant eye in the sky look at them",
			"image_ids": ["D3_IMG_000000"],
			"rel_ins_ids": []
		},
		"D3_INS_000001": {
			"instruction": "This is an AI generated image, can you infer what's the prompt behind this image?",
			"answer": "photography of a a soccer stadium on the moon, players are dressed as astronauts",
			"image_ids": ["D3_IMG_000001"],
			"rel_ins_ids": []
		}...
    }
}
```

Note that the `image_ids` is the key of the `DallE3_images.parquet` file, you can use the `image_ids` to index the `base64` string of the image.

## DallE3_images.parquet

```
import pandas as pd
images = "./DallE3_images.parquet"
image_parquet = pd.read_parquet(images)

image_parquet.head()
	                                                            base64
D3_IMG_000000	/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQEBAQEBAQ...
D3_IMG_000001	/9j/4AAQSkZJRgABAQEASABIAAD/5FolU0NBTEFETwAAAg...
```


Note that before September, we mainly use `images.json` to store the `key:base64_str` pairs, but we found it causes too much CPU memory during decoding large json files. So we switch to parquet, the parquet file is the same as previous json file and you can use the script to convert it from json to parquet.

```python
json_file_path = "LA.json"
with open(json_file_path, "r") as f:
    data_dict = json.load(f)
    
df = pd.DataFrame.from_dict(resized_data_dict, orient="index", columns=["base64"])
parquet_file_path = os.path.join(
    parquet_root_path, os.path.basename(json_file_path).split(".")[0].replace("_image", "") + ".parquet"
)
df.to_parquet(parquet_file_path, engine="pyarrow")
```
