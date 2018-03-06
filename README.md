# image_proc
**A repository containing various source codes for conventional image processing**

### Repository Overview:

[camera_calibration.py](./camera_calibration.py):contains an API which tries to minic the MATLAB's camera calibration app functionality. This API is a thin wrapper around the opencv's camera calibration functionalities.

[utils.py](./utils.py): contains various utility scripts

[examples](./examples): A diretory containing various examples


### Camera_Calibration_API:

#### Introduction:
The Camera Calibration API is a wrapper around the opencv's camera calibration functionalities. This tries to mimic the MATLAB camera calibration app's functionality in `python`. The API supports all the 3 calibration patterns supported by opencv namely: **Chessboards**, **Asymmetric circular grids** and **Symmetric circular grids.**  The API by default runs on 4 threads for speedup. The speed-up may not be marginal in the case of **chessboard** calibration because in most cases the bottle neck will be a single chessboard image (run on a single core) which the algorithm takes time to detect.

#### Dependencies:
* `works in both python-3 and python-2`
* `opencv (Tested in version 3.3.0)`
* `numpy`
* `matplotlib`
* `pickle`
* `argparse`
* `glob`
* `pickle`
* `multiprocessing`
* `os`
* `pandas`

#### Example:
Examples to use the Camera_Calibration_API() for calibration using chessboard, symmetric circular grids and asymmetric circular grids can be found in the [examples](./examples/example_notebooks) folder

#### Features:
* Supports all the 3 calibration patterns supported by opencv : **Chessboards**, **Asymmetric circular grids** and **Symmetric circular grids.**
* Additionally a **custom** calibration pattern can also be implemented. (Look at the next section for how to calibrate using custom pattern.)
* Visualizes the **Reprojection error plot**
* Ability to **Recalibrate** the camera by neglecting the images with very high reprojection errors.
* **Camera centric and Pattern centric** views can be visualized using the `visualize_calibration_boards` method after calibration. 
* `Blob detection parameters` for detecting asymmetric and symmetric circular grids can be accessed and modified via the **Camera_Calibration_API's object** prior to calling the `calibrate_camera` method
* Also has `terminal` support with **minimal control** on the variables. Use it as an importable module for better control over the variables 
* Can also be easily extended to support other unimplemented calibration patterns

#### Using custom calibration board with the Camera_Calibration_API.

So you want to extend the API for a custom calibration pattern? Well... OK! Just follow the follow the steps below

* The `calibrate_camera` accepts two additional arguments called `custom_world_points_function` and `custom_image_points_function`.
* You must implement the above two custom methods and pass it as an argument to the `calibrate_camera` method


        
##### custom_world_points_function(pattern_rows,pattern_columns):
    
* This function is responsible for calculating the 3-D world points of the given custom calibration pattern.
* Should take in two keyword arguments in the following order: Number of rows in pattern(int), Number of columns in pattern(int)
* Must return only a single numpy array of shape (M,3) and type np.float32 or np.float64 with M being the number of control points of the custom calibration pattern. The last column of the array (z axis) should be an array of 0
* The distance_in_world_units is not multiplied in this case. Hence, account for that inside the function before returning
* The world points must be ordered in this specific order : row by row, left to right in every row

##### custom_image_points_function(img,pattern_rows,pattern_columns):
    
* This function is responsible for finding the 2-D image points from the custom calibration image.
* Should take in 3 keyword arguments in the following order: image(numpy array),Number of rows in pattern(int), Number of columns in pattern(int)
* This must return 2 variables: return_value, image_points
* The first one is a boolean Representing whether all the control points in the calibration images are found
* The second one is a numpy array of shape (N,2) of type np.float32 containing the pixel coordinates or the image points of the control points. where N is the number of control points.
* This function should return True only if all the control points are detected (M = N)
* If all the control points are not detected, fillup the 2-D numpy array with 0s entirely and return with bool == False.
* The custom image points must be ordered in this specific order: : row by row, left to right in every row

**NOTE: 'Custom' pattern is not supported when accessed from terminal**



#### Supported Calibration patterns (rows x columns) bydefault:

##### Chessboard or Checkerboard pattern (6 x 9):
![chessboard](https://raw.githubusercontent.com/LongerVision/OpenCV_Examples/master/markers/pattern_chessboard.png)

##### Asymmetrical circular grid/pattern (4 x 11):
![Asymmetric circular grid](https://raw.githubusercontent.com/LongerVision/OpenCV_Examples/master/markers/pattern_acircles.png). 

#### NOTE for calibrating using Asymmetric circular grid:
* The code assumes that each asymmetric circle is placed at half the `distance_in_world_units` in both (x,y) from each other.

* The `distance_in_world_units` is specified as the distance between 2 adjacent circle centers at the **same y coordinate** 

* The above is a **4 x 11 (r x c)** asymmetrical circular grid. 

* If you are using the same orientation as the above, Then this orientation is termed as **double_count_in_column** which is by default set to `True`. 

* If you are using an orientation which is 90deg to the above orientation **11 x 4 (r x c)** then the `double count` is along the **rows**. In this case, set `object.double_count_in_column = False` prior to calling   `object.calibrate_camera` method.

##### Symmetric circular grid/pattern (7 x 6):
![Symmetrical circular pattern](http://answers.opencv.org/upfiles/13785495544653926.jpg)
