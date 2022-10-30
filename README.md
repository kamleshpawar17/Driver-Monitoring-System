# Driver State Monitoring System (DSMS)

This repo contains a project for DSMS using computer vision algorithms.

## To run application
```
python3 run_dsms.py
```
```run_dsms.py``` takes the following arguments:

```-m```: integer value, 1: to use python multiprocessing for make use of all CPU cores; 0: mean do not use multiprocessing.

```-np```: integer value equal to the number of parallel processes to use in case of multiprocessing.

```-c```: integer value, 1: to run a calibration to find a few parameters for different warnings; 0: do not run calibration.

```-sav```: integer value, 1: to save a video clip of the detected driver distraction events; 0: do not save the distraction events.

```config.yaml``` has many options (see comments in the file) to tune DSMS to optimize accuracy/speed.

## Dependencies
You should have OpenCV python installed and working to run this application.
Also make sure that you are able to access camera with opencv prior to running this application
```
opencv-python
imutils
dlib
pyyaml
```

## Note: 
Documentation for this project is in progress and will be updated here.

https://user-images.githubusercontent.com/32892726/198876465-5b6bdb4e-00ce-49ff-8090-f98c342f442c.mp4


