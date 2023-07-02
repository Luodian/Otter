# Convert It

This guide provides detailed instructions on how to convert various datasets from their public sources to our required format, including LLaVA-In-Context, Dense Captions, Visual Storytelling, TV Captions, Scene Navigation, Spot The Difference, and EGO4D. By following the specified steps, users can easily set up on these datasets. The output for each dataset will be saved in a corresponding JSON file named `<dataset_name>.json` in the `output` folder.

## LLaVA-In-Context

Download the [coco2017](https://cocodataset.org/#download) images (coco2014 might also be work), put the images in a folder with the path `<image_root>`. Download the [meta](https://drive.google.com/file/d/1iNHe8BUOALEdzuhRQ0ow0CGTvuD7JvQL/view?usp=sharing) for the training image ids, put the meta file at the path `<meta>`.

The folder structure should be like this:

```plain
<image_root>/
    annotations/
    val2017/
    train2017/
        000000498792.jpg
        XXXXXXXXXXXX.jpg
    ...
```

Run the following command (the `--num_threads` is optional):

```bash
python main.py --name=2d.Llava --image_path=<meta>  --image_root=<image_root>/train2017 [--num_threads=<num_threads>]
```

The output will be saved in `output/LA.json`.


## Dense Captions

Download the [Dense Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) videos in [ActivityNet](http://activity-net.org/challenges/2016/download.html#c3d), put the videos in a folder with the path `<image_path>` 

The folder structure should be like this:

```plain
<image_path>/
    <video_id>.mp4
    ...
```

Run the following command:

```bash
python main.py --name=video.DenseCaptions --image_path=<image_path> [--num_threads=<num_threads>]
```

The output will be saved in `output/DC.json`.

## Visual Storytelling

Download the [Visual Storytelling Dataset](https://visionandlanguage.net/VIST/dataset.html) and extract the `train.story-in-sequence.json` to a path, let `<json_path>` be the path of the json file, and run the following command:

```bash
python main.py --name=video.VisualStoryTelling --image_path=<json_path> [--num_threads=<num_threads>]
```

The output will be saved in `output/VST.json`.

## TV Captions

Download the [TV Captions video frames (3FPS)](https://tvqa.cs.unc.edu/download_tvqa.html#tvqa-download-4) and extract the `zip` to a path, let `<image_path>` be the path of the extracted folder.

The folder structure should be like this:

```plain
<image_path>/
    bbt_frames/
        ...
    castle_frames/
        ...
    house_frames/
        ...
    met_frames/
        ...
```

Run the following command:

```bash
python main.py --name=video.TVCaptions --image_path=<image_path> [--num_threads=<num_threads>]
```

The output will be saved in `output/TV.json`.

## Scene Navigation

Download the ScanNet v2 dataset from the [official website](http://www.scan-net.org/), let `<image_path>` be the path of the dataset

The folder structure should be like this:

```plain
<image_path>/
    scene0000_00/
        color/
            000000.jpg
            ...
        ...
    ...
```


Run the following command:

```bash
python main.py --name=3d.SceneNavigation --image_path=<image_path> [--num_threads=<num_threads>]
```

The output will be saved in `output/SN.json`.

## Spot The Difference (Subtle Difference Version)

Download the Spot The Difference Dataset from [Google Drive](https://drive.google.com/file/d/1OVb4_3Uec_xbyUk90aWC6LFpKsIOtR7v/view?usp=sharing), let `<image_path>` be the path of the dataset.

The folder structure should be like this:

```plain
<image_path>/
    <image>.jpg
    ...
```

Run the following command:

```bash
python main.py --name=change.SpotTheDifference --image_path=<image_path> [--num_threads=<num_threads>]
```

The output will be saved in `output/SD.json`.

## Spot The Difference (COCO General Difference Version)

Download the COCO 2017 train dataset from [COCO website](http://images.cocodataset.org/zips/train2017.zip), let `<image_path>` be the path of the dataset.

The folder structure should be like this:

```plain
<image_path>/
    <image>.jpg
    ...
```

Run the following command:

```bash
python main.py --name=change.CocoGeneralDifference --image_path=<image_path> [--num_threads=<num_threads>]
```

The output will be saved in `output/CGD.json`.

## EGO4D

Download the [EGO4D dataset](https://ego4d-data.org/#download), let `<image_path>` be the path of the dataset.

The folder structure should be like this:

```plain
<image_path>/
    <videos>.mp4
    ...
```

Run the following command:

```bash
python main.py --name=fpv.EGO4D --image_path=<image_path> [--num_threads=<num_threads>]
```

The output will be saved in `output/E4D.json`.
