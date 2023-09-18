# Face Recognition Software 

### Introduction 
This is a face recognition software built on python using face_recognition library, It works by getting images provided and then training its model inorder to identify a person face, It compares faces inorder to get the perfect match

### Installation
You'll often come across a problem of installing face_recognition, so I'm gonna guide

1. Begin by [downloading cmake](https://github.com/Kitware/CMake/releases/download/v3.27.5) from the cmake website, the link:  and make sure to add it to environment variables by ticking the option in the installation

2. Then install [python 3.8](https://www.python.org/ftp/python/3.8.10) on your system, this is because the dlib(a library that helps face_recognition run smoothly) is only supported on python 3.8 and below

3. Go to (https://github.com/RvTechiNNovate/face_recog_dlib_file) and then download the folder and then run 
    ```
    pip install dlib-19.19.0-cp37-cp37m-win_amd64.whl
    ```
    If It doesn't work then, try doing the same as above with the other file in the folder

4. After that all, install the requirements in the requirements.txt file, run this command
    ```
    pip install -r requirements.txt  
    ```
5. After all this, then run the following command, which is running the `main.py` file
    ```bash 
    pip install main.py
    ```
You link the images by appending a dict in the img_data property as below (If you change the folder name containing the images, change the `self.IMAGE_PATH`)

```python
self.IMAGE_PATH = "<image_folder_path_name>"

self.img_data = [
    {"name": "Name of face it belongs To", "image": "<img_path>"} # Don't include the name of the folder on the image path
]
```
Since the `self.IMAGE_PATH` will concatenated to the `self.img_data`

### By Ismael Swaleh (plexantech)