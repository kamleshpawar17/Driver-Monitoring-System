# Driver State Monitoring System (DSMS)

This repo contains a project for DSMS using computer vision algorithms.


https://user-images.githubusercontent.com/32892726/147677418-da8aff33-cbfe-4a9c-abc1-88018e8dc022.mp4


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
You should have OpenCV python working to run this application
```
opencv-python (4.0.0.1)
imutils
dlib
```

## Note: 
Documentation for this project is in progress and will be updated here.

