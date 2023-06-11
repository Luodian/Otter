<p align="center" width="100%">
<img src="./docs/mimicit_logo.png"  width="80%" height="80%">
</p>

## 🌳 MIMIC-IT Overview

High-quality instructions are essential for the zero-shot performance of large language models on interactive natural language tasks. For interactive vision-language tasks involving intricate visual scenes, a large quantity of diverse and creative instructions should be imperative to tune vision-language models (VLMs). Nevertheless, the current availability of vision-language instructions in terms of quantity, diversity, and creativity remains limited, posing challenges to the generalization of interactive VLMs. Here we present **MIMIC-IT**, a dataset comprising 2.8M multi-modal instructions-response pairs based on images and videos. Each instruction-response pair is accompanied by multi-modal in-context information, forming conversational contexts aimed at empowering VLMs in perception, reasoning, and planning. The instruction-response collection process, dubbed as **Syphus**, is scaled using an automatic annotation pipeline that combines human expertise with GPT's capabilities.

MIMIC-IT covers a vast array of real-life scenarios that empower Vision-Language Models (VLMs) to not only comprehend general scenes, but also to reason about context and astutely differentiate between observations. MIMIC-IT also enables the application of egocentric visual assistant model that can serve that can answer your questions like **Hey, Do you think I left my keys on the table?**. In addition to English, MIMIC-IT is also multilingual, supporting Chinese, Korean, Japanese, German, French, Spanish, and Arabic, thereby allowing a larger global audience to altogether enjoy from the convenience brought about by advancements in artificial intelligence.

<p align="center" width="100%">
<img src="https://i.postimg.cc/4x66gHhw/mimic-it.jpg"  width="80%" height="80%">
</p>

## Dataset Statistics

| **Visual Sources (Scenes)** | **In-context** | **#Clips/Images** | **#Uni. Inst.** | **#Instances** |  
|---------------------------|----------------|------------------|-----------------|----------------|  
| COCO (General) | lang./vis. | - / 81K | 261K | 345K |  
| SD~(Surveillance) | lang./vis. | - / 9K | 10K | 15K  |  
| SN~(Indoor Ego.) | lang./vis. | - / 0.5K | 4.8K | 6K  |  
| DC~(General) | lang./vis. | 16K / 1M | 40K | 62K  |  
| VIST~(Story) | lang./vis. | - / 16K | 32K | 33K  |  
| TVC~(TV) | lang./vis. | 86K / 577K | 86K | 92K  |  
| E4D~(General Ego.) | lang./vis. | 400K / 6.4M | 1.8M | 2.4M  |  
| Total | lang./vis. | 502K / 8.1M | 2.2M | 2.8M |  

## Download Links

The initial release includes LA and DC instruction-response pairs for the MIMIC-IT dataset. We plan to release additional datasets with a larger number of instruction pairs and more information after further examination.

We are contacting the image sources (those public datasets we used) to ask if we can directly release their image/video data in our Otter training format (base64 format within a large JSON file), we will put these data in following link if there would not be any legal/license issue. 

This process may take some time. If you are interested in using this data, please leave an issue in this repository or email drluodian@gmail.com, and we will keep you updated.

Additionally, we are in the process of providing the scripts used to convert public dataset images and extract specific frames from corresponding videos into the MIMIC-IT input format. This will help map the original dataset to our annotations UUIDs (e.g. from COCO's `000000215677.jpg` -> ours `LA_00_IMG_000000215677`).


| Scenes | Images/Videos | Size | Annotations | Size |
| :--- | :---: | :---: | :---: | :---: |
| **LA In-context** | Processing | 5.2GB |[link](https://entuedu-my.sharepoint.com/:u:/r/personal/libo0013_e_ntu_edu_sg/Documents/MIMIC-IT-Release/LA_instructions.json.zip?csf=1&web=1&e=SvaKh3) | 269.3MB |
| **Dense Caption** | Processing | 86.4GB |[link](https://entuedu-my.sharepoint.com/:u:/r/personal/libo0013_e_ntu_edu_sg/Documents/MIMIC-IT-Release/DC_instructions.json.zip?csf=1&web=1&e=jM4gGB) | 269.1MB | 
| **TV Caption** | Processing | 17.0GB | Cleaning | 55.6MB |
| **Visual Story Telling** | Processing | 16.2GB |Cleaning | 33.4MB |
| **Scene Navigation (Indoor Event Planning)** | Processing | 2.3GB |Cleaning | 7.6MB |
| **Spot The Difference (COCO's General Difference)** | Processing | 5.2GB |Cleaning | 80.5MB |
| **Spot The Difference (Subtle Difference)** | Processing | 3.1GB |Cleaning | 5.0MB |
| **EGO4D** | Processing | ~500GB |Cleaning | 3.2GB |

The data is available on [NTU-Onedrive](https://entuedu-my.sharepoint.com/:f:/g/personal/libo0013_e_ntu_edu_sg/Eo9bgNV5cjtEswfA-HfjNNABL6Eazh7Fm1dX5VlI0Bqsrg?e=cMvZ1a). The JSON files are compressed into ZIP files to save space. After downloading, unzip the files and verify the MD5 checksums to ensure their integrity. The MD5 Checksums for the released annotations are:

1. LA_instructions.json (json not the zip file) -> f9bc559391d15727b35f3df306b12e31
2. DC_instructions.json -> bb0d1f9f7d100c99869f79d13b3a3beb

The MIMIC-IT dataset is stored in the following format:
```json
{  
  "meta": {  
    "version": "0.0.1",  
    "time": "2023-06",  
    "author": "ntu"  
  },  
  "data": {  
    "DC_04_INS_00001": {  
      "instruction": "Who is the main focus of the video?",  
      "answer": "The main focus of the video is a police officer riding a horse down the street.",  
      "image_ids": [  
        "DC_04_IMG_v_N1c3C_Npr-E_0000",  
        "DC_04_IMG_v_N1c3C_Npr-E_0001",  
        ...  
        "DC_04_IMG_v_N1c3C_Npr-E_0067"  
      ],  
      "rel_ins_ids": [  
        "DC_04_INS_00002",  
        "DC_04_INS_00003",  
        ...  
        "DC_04_INS_00008"  
      ]  
    },  
    ...  
  }  
}  
```

This JSON file includes a meta object with version, time, and author information. The data object contains instruction-response pairs, each with a unique identifier (e.g., "DC_04_INS_00001"). Each pair consists of an instruction, an answer, an array of associated image IDs, and an array of related instruction IDs (which can be arranged as in-context examples).

## Multilingual Instruction-Response Pairs

We will release  multilingual instruction-response pairs in the following languages:

<p align="center" width="100%">
<img src="https://i.postimg.cc/nLwQtfZ1/multilingual.png"  width="80%" height="80%">
</p>

## Syphus Overview

<p align="center" width="100%">
<img src="https://i.postimg.cc/RCGp0vQ1/syphus.png"  width="80%" height="80%">
</p>

Syphus, an automated pipeline for generating high-quality instruction-response pairs in multiple languages. Building upon the framework proposed by LLaVA, we utilize ChatGPT to generate instruction-response pairs based on visual content. To ensure the quality of the generated instruction-response pairs, our pipeline incorporates system messages, visual annotations, and in-context examples as prompts for ChatGPT. System messages define the desired tone and style of the generated instruction-response pairs, while visual annotations provide essential image information such as bounding boxes and image descriptions. In-context examples assist ChatGPT in learning within the context.  During cold-start stage, in-context examples are collected by prompting ChatGPT solely through system messages and visual annotations, employing a heuristic approach. This stage concludes only when a satisfactory in-context examples are identified. In step 4, once the instruction-response pairs are obtained, the pipeline expands them into Chinese (zh), Japanese (ja), Spanish (es), German (de), French (fr), Korean (ko), and Arabic (ar).

## Syphus on your own dataset

We provide source code of the framework in [syphus](Syphus) folder. You can use it to generate instruction-response pairs on your own dataset following the steps below.

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
python  main.py --name YourDataset.your_dataset --num_threads 4
```

## 📝 Citation
