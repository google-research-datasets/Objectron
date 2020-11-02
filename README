**This is not an officially supported Google product.**

# Objectron Dataset

[The Objectron website!](https://www.objectron.dev)

The Objectron dataset is a collection of short, object-centric video clips each of which is accompanied by AR session metadata that includes camera poses, sparse point-clouds and characterization of the planar surfaces in the surrounding environment. In each video, the camera moves around the object, capturing it from different angles. The data also contain manually annotated 3D bounding boxes for each object, which describe the object’s position, orientation, and dimensions. The dataset consists of 15K annotated video clips supplemented with over 4M annotated images in the following categories: bikes, books, bottles, cameras, cereal boxes, chairs, cups, laptops, and shoes. In addition, to ensure geo-diversity, our dataset is collected from 10 countries across five continents.


# Dataset Format
The data is stored in the [objectron bucket](https://storage.googleapis.com/objectron) on Google Cloud storage, and includes the following assets:
- The video sequences (located in /videos/class/batch-i/j/video.MOV files)
- The annotation labels containing the 3D bounding boxes for objects. The annotation protobufs are located in /videos/class/batch-i/j/geometry.pbdata files. They are formatted using the [object.proto](https://github.com/google-research-datasets/Objectron/blob/master/objectron/schema/object.proto). See [example] on how to parse the annotation files.
- AR metadata (such as camera poses, point clouds, and planar surfaces). They are based on [a_r_metadata_capture.proto](https://github.com/google-research-datasets/Objectron/blob/master/objectron/schema/a_r_metadata_capture.proto). See [example]() on how to parse these files.
- Processed dataset: sharded and shuffled tf.records of the annotated frames, in tf.example format. These are used for creating the input data pipeline to your models. These files are located in /v1/records_sharded/class/
- Supporting scripts to run evaluation based on the metric described above. 
- Supporting scripts to load the data into Tensorflow, Jax and Pytorch and visualize the dataset, including “Hello World” examples. 
- Supporting Apache Beam jobs to process the datasets on Google Cloud infrastructure.
- The index of all available samples, as well as train/test splits for easy access and download.

This repository provides the required schemas and tools to parse the dataset. 


