# Driver State Monitoring System (DSMS)

This repo contains a project for DSMS using computer vision algorithms.


## To run code
```
python3 run_dsms.py
```
```run_dsms.py``` takes the following arguments:

```-m```: integer value, 1: means use python multiprocessing for make use of all CPU cores; 0: mean do not use multiprocessing.

```-np```: integer value equal to the number of parallel processes to use in case of multiprocessing.

```-c```: integer value, 1: means run a calibration to find a few parameters for different warnings; 0: do not run calibration.

```-sav```: integer value, 1: means save a video clip of the detected driver distraction events; 0: do not save the distraction events.


## Dependencies
You should have OpenCV python working to run.
```
opencv-python (4.0.0.1)
imutils
dlib
```

## Note: 
Documentation for this project is in progress and will be updated here.

