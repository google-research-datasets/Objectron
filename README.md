
<div align="center">

# Objectron Dataset

**Objectron is a dataset of short object centric video clips with pose annotations.**

<img src="docs/images/objectron_samples.gif" width="600px">

---
<p align="center">
  <a href="https://www.objectron.dev">Website</a> •
  <a href="#dataset-format">Dataset Format</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#license">License</a>
</p>

</div>



The Objectron dataset is a collection of short, object-centric video clips, which are accompanied by AR session metadata that includes camera poses, sparse point-clouds and characterization of the planar surfaces in the surrounding environment. In each video, the camera moves around the object, capturing it from different angles. The data also contain manually annotated 3D bounding boxes for each object, which describe the object’s position, orientation, and dimensions. The dataset consists of 15K annotated video clips supplemented with over 4M annotated images in the following categories: `bikes, books, bottles, cameras, cereal boxes, chairs, cups, laptops`, and `shoes`. In addition, to ensure geo-diversity, our dataset is collected from 10 countries across five continents. Along with the dataset, we are also sharing a [3D object detection solution](https://google.github.io/mediapipe/solutions/objectron) for four categories of objects — shoes, chairs, mugs, and cameras. These models are trained using this dataset, and are released in [MediaPipe](https://mediapipe.dev/), Google's open source framework for cross-platform customizable ML solutions for live and streaming media.

## Key Features
- 15000 annotated videos and 4M annotated images
- All samples include high-res images, object pose, camera pose, point-cloud, and surface planes.
- Ready to use examples in various tf.record formats, which can be used in Tensorflow/PyTorch.
- Object-centric multi-views, observing the same object from different angles.
- Accurate evaluation metrics, like 3D IoU for oriented 3D bounding boxes.


## Dataset Format
The data is stored in the [objectron bucket](https://storage.googleapis.com/objectron) on Google Cloud storage. Check out the [Download Data](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Download%20Data.ipynb) notebook for a quick review of how to download/access the dataset. The following assets are available:

- The video sequences (located in `/videos/class/batch-i/j/video.MOV` files)
- The annotation labels containing the 3D bounding boxes for objects. The annotation protobufs are located in `/videos/class/batch-i/j/geometry.pbdata` files. They are formatted using the [object.proto](https://github.com/google-research-datasets/Objectron/blob/master/objectron/schema/object.proto). See [example] on how to parse the annotation files.
- AR metadata (such as camera poses, point clouds, and planar surfaces). They are based on [a_r_capture_metadata.proto](https://github.com/google-research-datasets/Objectron/blob/master/objectron/schema/a_r_capture_metadata.proto). See [example](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/objectron-geometry-tutorial.ipynb) on how to parse these files.
- Processed dataset: sharded and shuffled `tf.records` of the annotated frames, in tf.example format and videos in `tf.SequenceExample` format. These are used for creating the input data pipeline to your models. These files are located in `/v1/records_shuffled/class/` and `/v1/sequences/class/`.
- Supporting scripts to run evaluation based on the 3D IoU metric.
- Supporting scripts to load the data into Tensorflow, Jax and Pytorch and visualize the dataset, including “Hello World” examples.
- Supporting Apache Beam jobs to process the datasets on Google Cloud infrastructure.
- The index of all available samples, as well as train/test splits for easy access and download.

Raw dataset size is 1.9TB (including videos and their annotations). Total dataset size is 4.4TB (including videos, records, sequences, etc.). This repository provides the required schemas and tools to parse the dataset.

| class   | [bike](https://github.com/google-research-datasets/Objectron/blob/master/index/bike_annotations) | [book](https://github.com/google-research-datasets/Objectron/blob/master/index/book_annotations) | [bottle](https://github.com/google-research-datasets/Objectron/blob/master/index/bottle_annotations) | [camera](https://github.com/google-research-datasets/Objectron/blob/master/index/camera_annotations) | [cereal_box](https://github.com/google-research-datasets/Objectron/blob/master/index/cereal_box_annotations) | [chair](https://github.com/google-research-datasets/Objectron/blob/master/index/chair_annotations) | [cup](https://github.com/google-research-datasets/Objectron/blob/master/index/cup_annotations)  | [laptop](https://github.com/google-research-datasets/Objectron/blob/master/index/laptop_annotations) | [shoe](https://github.com/google-research-datasets/Objectron/blob/master/index/shoe_annotations) |
|---------|------|------|--------|--------|------------|-------|------|--------|------|
| #videos | 476  | 2024 | 1928   | 815    | 1609       | 1943  | 2204 | 1473   | 2116 |
| #frames | 150k | 576k | 476k   | 233k   | 396k       | 488k  | 546k | 485k   | 557k |

## Tutorials
- [Downloading the dataset](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Download%20Data.ipynb)
- [Hello, World example: Loading examples in Tensorflow](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Hello%20World.ipynb)
- [Loading data in PyTorch](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Objectron_Pytorch_tutorial.ipynb)
- [Parsing raw annotation files](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Parse%20Annotations.ipynb)
- [Parsing the AR metadata](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/objectron-geometry-tutorial.ipynb)
- [Evaluating the model performance with 3D IoU](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/3D_IOU.ipynb)
- [SequenceExample tutorial](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/SequenceExamples.ipynb)
- [Training a NeRF model](https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Objectron_NeRF_Tutorial.ipynb)

## License
Objectron is released under [Computational Use of Data Agreement 1.0 (C-UDA-1.0)](https://github.com/microsoft/Computational-Use-of-Data-Agreement). A [copy](https://github.com/google-research-datasets/Objectron/blob/master/LICENSE) of the license is available in this repository.


## BibTeX
If you found this dataset useful, please cite our [paper](https://arxiv.org/abs/2012.09988).

```
@article{objectron2021,
  title={Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild with Pose Annotations},
  author={Adel Ahmadyan, Liangkai Zhang, Artsiom Ablavatski, Jianing Wei, Matthias Grundmann},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

**This is not an officially supported Google product.** If you have any question, you can email us at objectron@google.com or join our mailing list at objectron@googlegroups.com


