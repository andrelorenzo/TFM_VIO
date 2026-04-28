# PYTHON SCRIPT TOOLS

### CALIBRATION OF IMU USING ALLAN VARIANCE METHOD (HOW TO)

* Measurements are espected in m/s2 and rad/s
* --g<>-col <name> describes the name of the colum in the csv file
* --use-timestamp --timestamp-col time_ms instead qof --fs for automatic calculation of the frequency
* --csv is for input file
* --output is for output result file

```bash
    python script.py --csv datos.csv --gx-col gx --gy-col gy --gz-col gz --ax-col ax --ay-col ay --az-col az --fs 100 --output resultado.txt
```

This programm will calculate the N, B,K Allan variance constant for sensor modeling
This programm will calculate static bias for gyro and accel, based on positive +9.80665 on z axis


### CALIBRATION OF CAMERA USING CHECKBOARD PATTERN (HOW TO)

* --video = extract images of checkboard from video
* --max-frames = max captures from a video
* --frame-step = each n frames we take a picture

* --camera-id = for PC camera

```bash
    python cam_calib.py --images "calib/phone/*.jpg" --output calib/phone/calib.yaml --cols 8 --rows 6 --square-size 0.029 --show
```
This will generate a .yaml file with: 
* Width
* Height
* Intrinsic matrix in opencv format (K)
* Dist coeficients in opencv format (D)
* RMS value must be between 0.1 and 0.5 to be acceptable:
    * Excellent: < 0.2
    * Good: 0.2 - 0.5
    * Acceptable: < 1.0
    * Bad: > 1.0 (blurry images, improper board handling, high lens distortion, corners poorly detected, nº of images is too little)


### SHOW TRAYECTORY FROM CSV 

* Show ground truth: gt
* Show position: est
* Show in 2D: --topdown

```bash
    python plot_tray.py datos.csv --series gt --topdown
```
