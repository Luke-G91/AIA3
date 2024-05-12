# Data directory

### To create a new model

1. add classifications directories to this folder to create a new model
2. add new directories for each class within the classification
3. populate each class with relevant png images
4. update the category variable in the notebook to access the new data

file structure should resemble the following:

```
data
│   README.md
│
└───classification-1
│   │
│   └───classification-1-class-1
│   │   │   image1.png
│   │   │   image2.png
│   │   │   ...
|   |
│   └───classification-1-class-2
│       │   image3.png
│       │   image4.png
│       │   ...
|
└───classification-2
│   │
│   └───classification-2-class-1
│   │   │   image5.png
│   │   │   image6.png
│   │   │   ...
|   |
│   └───classification-2-class-2
│       │   image7.png
│       │   image8.png
│       │   ...
```
