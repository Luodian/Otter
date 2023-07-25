<p align="center" width="100%">
<img src="https://i.postimg.cc/sxy8v9PS/mimicit-logo.png"  width="80%" height="80%">
</p>

- [🌳 MIMIC-IT Overview](#-mimic-it-overview)
- [Using MIMIC-IT Dataset](#using-mimic-it-dataset)
  - [Convert It](#convert-it)
  - [Download It](#download-it)
  - [Eggs](#eggs)
- [Syphus: the hero behind MIMIC-IT](#syphus-the-hero-behind-mimic-it)
  - [Syphus on your own dataset](#syphus-on-your-own-dataset)
- [Multilingual Instruction-Response Pairs](#multilingual-instruction-response-pairs)

## 🌳 MIMIC-IT Overview

MIMIC-IT offers a diverse and extensive dataset of 2.8M multimodal instruction-response pairs, designed to enhance the performance of Vision-Language Models (VLMs) in real-life scenarios, enabling VLMs to excel in perception, reasoning, and planning while also catering to a multilingual audience. 

MIMIC-IT enables the application of egocentric visual assistant model that can serve that can answer your questions like **Hey, Do you think I left my keys on the table?**. Harness the power of MIMIC-IT to unlock the full potential of your AI-driven visual assistant and elevate your interactive vision-language tasks to new heights.

MIMIC-IT provides multilingual instructions, supporting English, Chinese, Korean, Japanese, German, French, Spanish, and Arabic, thereby allowing a larger global audience to altogether enjoy from the convenience brought about by advancements in artificial intelligence.

<p align="center" width="100%">
<img src="https://i.postimg.cc/4x66gHhw/mimic-it.jpg"  width="100%" height="100%">
</p>



## Using MIMIC-IT Dataset

We have integrated the MIMIC-IT dataset into the Hugging Face dataset. You can download and utilize the MIMIC-IT dataset from [here](https://huggingface.co/datasets/pufanyi/MIMICIT).

You can following the steps to obtain the MIMIC-IT dataset. Each task (e.g. `DC`, `LA`) in MIMIC-IT is composed of three parts, including:
1. `xx.json` file: the images in base64 format.
2. `xx_instructions.json` file: the instruction-response pairs (also includes image ids and related instructions ids for each instruction-response pair) for each task.
3. `xx_train.json` file: the customized related instruction-response pairs for each instruction.

The following steps will introduce you how to gather them together.

### Convert It

You may need to refer to the [Convert-It](./convert-it/README.md) to convert the image sources from public dataset to the format of `xx.json`. If you find it is hard to download from the original image sources, you can refer to the [Eggs](#eggs) section to seek help there.

### Download It

You can download the `instructions.json` and `train.json` files, from our provided [OneDrive folder](https://entuedu-my.sharepoint.com/:f:/g/personal/libo0013_e_ntu_edu_sg/Eo9bgNV5cjtEswfA-HfjNNABiKsjDzSWAl5QYAlRZPiuZA?e=M9isDT).

| Tasks/Scenes | Zip File MD5 Checksum | Unzipped File Size |
| :--- | :---: | :---: |
| **LA-In-Context**  | fdc2427451bcfd8a04bab7a1c2305259 | 338 MB |
| **DC** | 0d373a5f511fd1d9f03ac81bb12e04fe | 171 MB | 
| **TVC** | ea16ec0ef7f35e810e0920e85ed467af | 166 MB |
| **VST** | 988569e39aaa24da0df547644514b0d4 | 32 MB |
| **SN** | 1c4751c5b2c0bcaaeb94dbc5fb39e7a6 | 8 MB |
| **SD (General Diff)** | 7fd998c10baaba9c6e39b66761a456a0 | 8.1 MB |
| **SD (Subtle Diff)** | 5175198daebb997672a21307e8b18a96 | 5 MB |
| **E4D (1st Part)** | 504b779dbc852c943adbe7862d6924d7 | 710 MB/3.2 GB |

After downloading, unzip the files and place them in the `mimicit_data` folder. The folder structure should be as follows:

```bash
mimicit_data/DC/DC_instructions.json
mimicit_data/DC/DC_train.json
```
The `DC_instructions.json` includes a meta object with version, time, and author information. The data object contains instruction-response pairs, each with a unique identifier (e.g., "DC_INS_00001"). Each pair consists of an instruction, an answer, an array of associated image IDs, and an array of related instruction IDs (which can be arranged as in-context examples).
```json
{   
    "meta":{"version":"0.0.1","time":"2023-06","author":"ntu"},
    "data": {
        "DC_INS_00001": {
            "instruction":"Who is the main focus of the video?",
            "answer":"The main focus of the video is a police officer riding a horse down the street.",
            "image_ids":["DC_IMG_v_N1c3C_Npr-E_0000","DC_IMG_v_N1c3C_Npr-E_0001","DC_IMG_v_N1c3C_Npr-E_0002",..."],
            "rel_ins_ids":["DC_INS_00002","DC_INS_00003","DC_INS_00004","DC_INS_00005","DC_INS_00006","DC_INS_00007","DC_INS_00008"]
        },
    }
    ...
}  
```

The `DC_train.json` contains instructions IDs and their associated related instruction IDs. Each instruction is associated with its related instructions. We provide it for more flexibly define each instruction's related instructions. It serves for different in-context learning objectives. In default, the related instructions ids are from `rel_ins_ids` at `DC_instructions.json`. But you can define your own related instructions ids for each instruction by just creating your own `DC_train.json` file.

```json
{
    "DC_INS_00001": ["DC_INS_00002", "DC_INS_00003", "DC_INS_00004", "DC_INS_00005", "DC_INS_00006", "DC_INS_00007", "DC_INS_00008"],
    ...
}
```

### Eggs

Things could be tricky since some image/video sources are not easy to get the access to download them. We also provide the converted `xx.json` files for you to download directly. You need to agree the same terms and conditions as the original dataset, as well as recognize and appreciate the contributions made by these data sources. Please refer to [Google form](https://forms.gle/kYXPVDiscNvKhv6b6) to apply for the access to download the converted `xx.json` files.

Access to the images provided is exclusively for contributing positively to the academic research community. Usage of these images is required to align with all pertinent licenses governing their distribution. By engaging with this content, you are making a commitment to utilize the images solely for the stated non-commercial purposes and to comply with the stipulations of the original licenses. 

Moreover, filling in and submitting the form with your verifiable name, institutional affiliation, and signature serves as your binding acknowledgment and affirmation to uphold these terms and conditions with integrity.

<!-- ### Dataset Statistics
[Title](https://forms.gle/kYXPVDiscNvKhv6b6)
| **Visual Sources (Scenes)** | **In-context** | **#Clips/Images** | **#Uni. Inst.** | **#Instances** |  
|---------------------------|----------------|------------------|-----------------|----------------|  
| COCO (General) | lang./vis. | - / 81K | 261K | 345K |  
| SD (Surveillance) | lang./vis. | - / 9K | 10K | 15K  |  
| SN (Indoor Ego.) | lang./vis. | - / 0.5K | 4.8K | 6K  |  
| DC (General) | lang./vis. | 16K / 1M | 40K | 62K  |  
| VIST (Story) | lang./vis. | - / 16K | 32K | 33K  |  
| TVC (TV) | lang./vis. | 86K / 577K | 86K | 92K  |  
| E4D (General Ego.) | lang./vis. | 400K / 6.4M | 1.8M | 2.4M  |  
| Total | lang./vis. | 502K / 8.1M | 2.2M | 2.8M |   -->
## Syphus: the hero behind MIMIC-IT

<p align="center" width="100%">
<img src="https://i.postimg.cc/RCGp0vQ1/syphus.png"  width="80%" height="80%">
</p>

Embracing Syphus, an automated pipeline that generates top-tier instruction-response pairs in various languages. 

Syphus builds on the LLaVA framework and uses ChatGPT to produce pairs based on visual content. It ensures quality by incorporating system messages for tone and style, visual annotations for essential image information, and in-context examples to assist ChatGPT in contextual learning. During the cold-start stage, in-context examples are collected using a heuristic approach with system messages and visual annotations. This stage concludes only when a satisfactory in-context examples are identified.

Finally, the pipeline expands the instruction-response pairs into languages like Chinese, Japanese, Spanish, German, French, Korean, and Arabic.

### Syphus on your own dataset

We provide source code of the framework in [syphus](./syphus/) folder. You can use it to generate instruction-response pairs on your own dataset following the steps below.

1. Configure openai key. Create the following environment variables in your system.

``` bash
export OPENAI_API_TYPE="xxx"
export OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxx"
export OPENAI_API_BASE="https://xxxx"
export OPENAI_API_VERSION="2023-03-15-xx"
export OPENAI_API_ENGINE="chatgpt0301"
```

1. Create a new file in `datasets` folder and name it as `your_dataset.py`.
2. Create a class named `YourDataset` in `your_dataset.py` and inherit from `abstract_dataset.AbstractDataset` class.
3. Implement `_load_query_inputs` methods in `YourDataset` class. This method should return a list of dict, each dict contains `id` and `sentences` keys.
   1. `id` is the unique id of each query
   2.  `sentences` is a string for each query input.
4.  Define `system_message`, `in-context example` in `prompts` folder.
5.  Define `_load_prefix` methods in `YourDataset` class. 
    1.  This method should return a dictionary. The keys of the dictionary are `system_message`, `in_context`.
    2.  The value of `system_message` is a string.
    3.  The value of `in_context` is a list of `{"role": role, "content": content_string}`.
6. You are done! Run the following command to generate instruction-response pairs on your own dataset.

``` bash
python  main.py --name YourDataset.your_dataset --num_threads 64
```

## Multilingual Instruction-Response Pairs

We will release  multilingual instruction-response pairs in the following languages:

<p align="center" width="100%">
<img src="https://i.postimg.cc/nLwQtfZ1/multilingual.png"  width="80%" height="80%">
</p>
