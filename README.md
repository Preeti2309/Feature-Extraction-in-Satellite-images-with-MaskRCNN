FEATURE EXTRACTION IN SATELLITE IMAGE USING MASK RCNN IN DEEP LEARNING

This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

The need for up-to-date information about earth’s surface is growing, as such, information provides a base for a large number of applications, including local, regional and global resources monitoring, land-cover and land-use change monitoring, geographical life development and environmental studies.  The data from remote sensing satellites provide opportunities to acquire information about land at varying resolutions and has been widely used for object studies. The importance of the exponential increase in the image data volume and multiple sensors and associated challenges on the development of object detection techniques are highlighted. Object detection is the task of detecting objects in an image in the form of a bounding box.  Challenges faced in object detection include non-trivial noisiness, blurring and lower resolutions compared to aerial images. Objects may overlap and have different textures or alternately, adjacent objects may have similar texture, which can prevent identification of contours separating them.  This task can either be carried out by a domain expert (manually) or automatically by employing machine learning. Machine learning techniques if implemented properly may give results better than a human and can also reduce the expenditure to a very large extent.To get a more accurate information about the object, more than a rectangle (bounding box), maybe a polygon which represents the object more tightly is preferred. But that’s still not the best way. The best way would be to assign each pixel inside the bounding box which actually has the object. This task is called as Instance segmentation, where the object instances are segmented.  This project implements Instance Segmentation using deep learning to extract geographical features from satellite maps. It’s called Mask R-CNN,assumes a basic understanding of deep learning and CNNs for object detection.  


The repository includes:

    Source code of Mask R-CNN built on FPN and ResNet101.
    Training code for MS COCO
    Pre-trained weights for MS COCO
    Jupyter notebooks to visualize the detection pipeline at every step
    ParallelModel class for multi-GPU training
    Evaluation on MS COCO metrics (AP)
    Example of training on your own dataset

