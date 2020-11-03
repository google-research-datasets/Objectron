
<div align="center">

# Objectron Dataset

**Objectron is a dataset of short object centeric video clips with pose annotations.**

---
<p align="center">
  <a href="https://www.objectron.dev">Website</a> •
  <a href="#key-features">Dataset Format</a> •
  <a href="#examples">Examples</a> •
  <a href="#licence">Licence</a>
</p>

</div>

<img src="docs/images/objectron_samples.gif" width="600px">

The Objectron dataset is a collection of short, object-centric video clips each of which is accompanied by AR session metadata that includes camera poses, sparse point-clouds and characterization of the planar surfaces in the surrounding environment. In each video, the camera moves around the object, capturing it from different angles. The data also contain manually annotated 3D bounding boxes for each object, which describe the object’s position, orientation, and dimensions. The dataset consists of 15K annotated video clips supplemented with over 4M annotated images in the following categories: bikes, books, bottles, cameras, cereal boxes, chairs, cups, laptops, and shoes. In addition, to ensure geo-diversity, our dataset is collected from 10 countries across five continents.

## Key Features
- 15000 annotated videos and 4M annotated images
- All samples include high-res images, object pose, camera pose, point-cloud, and surface planes.
- Ready to use examples in various tf.record format, which can be used in Tensorflow/PyTorch. 
- Object-centric multi-views, observing the same object from different angles.
- Accurate evaluation metrics, like 3D IoU for oriented 3D bounding boxes.


## Dataset Format
The data is stored in the [objectron bucket](https://storage.googleapis.com/objectron) on Google Cloud storage. Check out the [Downoad Data](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Download%20Data.ipynb) notebook for a quick review of how to download/access the dataset. The following assets are available:

- The video sequences (located in `/videos/class/batch-i/j/video.MOV` files)
- The annotation labels containing the 3D bounding boxes for objects. The annotation protobufs are located in `/videos/class/batch-i/j/geometry.pbdata` files. They are formatted using the [object.proto](https://github.com/google-research-datasets/Objectron/blob/master/objectron/schema/object.proto). See [example] on how to parse the annotation files.
- AR metadata (such as camera poses, point clouds, and planar surfaces). They are based on [a_r_metadata_capture.proto](https://github.com/google-research-datasets/Objectron/blob/master/objectron/schema/a_r_metadata_capture.proto). See [example]() on how to parse these files.
- Processed dataset: sharded and shuffled tf.records of the annotated frames, in tf.example format. These are used for creating the input data pipeline to your models. These files are located in /v1/records_sharded/class/
- Supporting scripts to run evaluation based on the metric described above. 
- Supporting scripts to load the data into Tensorflow, Jax and Pytorch and visualize the dataset, including “Hello World” examples. 
- Supporting Apache Beam jobs to process the datasets on Google Cloud infrastructure.
- The index of all available samples, as well as train/test splits for easy access and download.

This repository provides the required schemas and tools to parse the dataset. 

## Examples
- [Downloading the dataset](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Download%20Data.ipynb)
- [Hello, World example: Loading examples in Tensorflow](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Hello%20World.ipynb)
- [Loading data in PyTorch](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Objectron_Pytorch_tutorial.ipynb)
- [Parsing raw annotation files](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Parse%20Annotations.ipynb)
- [Evaluating the 3D IoU metric](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/3D_IOU.ipynb)

## License
Objectron is released under [Computational Use of Data Agreement 1.0 (C-UDA-1.0)](https://github.com/microsoft/Computational-Use-of-Data-Agreement). A [copy](https://github.com/google-research-datasets/Objectron/blob/master/LICENSE) of the license is available in this repository.


**This is not an officially supported Google product.**


