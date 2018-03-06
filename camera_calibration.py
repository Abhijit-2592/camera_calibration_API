#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:03:39 2018

@author: abhijit
"""

# Python 2/3 compatibility
from __future__ import print_function
# this is important else throws error
from mpl_toolkits.mplot3d import Axes3D 

from matplotlib import cm
from numpy import linspace

import numpy as np
import cv2 
import matplotlib.pyplot as plt
# local modules

# built-in modules
import os
from multiprocessing.dummy import Pool as ThreadPool
import argparse
from argparse import RawTextHelpFormatter
import glob
import pickle
import pandas as pd



class Camera_Calibration_API:
    """ A complete API to calibrate camera with chessboard or symmetric_circles or asymmetric_circles.
        also runs on multi-threads
    
    Constructor keyword arguments:
    pattern_type --str: One of ['chessboard','symmetric_circles,'asymmetric_circles','custom'] (No default)
    pattern_rows --int: Number of pattern points along row (No default)
    pattern_columns --int: Number of pattern points along column (No default)
    distance_in_world_units --float: The distance between pattern points in any world unit. (Default 1.0)
    figsize: To set the figure size of the matplotlib.pyplot (Default (8,8))
    debug_dir --str: Optional path to a directory to save the images  (Default None)
                                 The images include : 
                                 1.Points visulized on the calibration board
                                 2.Reprojection error plot
                                 3.Pattern centric and camera centric views of the calibration board
    term_criteria: The termination criteria for the subpixel refinement (Default: (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001))

    """
    
    def __init__(self,
                 pattern_type,
                 pattern_rows,
                 pattern_columns,
                 distance_in_world_units = 1.0,
                 figsize = (8,8),
                 debug_dir = None,
                 term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
                 ):
        
        pattern_types = ["chessboard","symmetric_circles","asymmetric_circles","custom"]
        
        assert pattern_type in pattern_types, "pattern type must be one of {}".format(pattern_types)
        
        self.pattern_type = pattern_type
        self.pattern_rows = pattern_rows
        self.pattern_columns = pattern_columns
        self.distance_in_world_units = distance_in_world_units
        self.figsize = figsize
        self.debug_dir = debug_dir
        self.term_criteria = term_criteria
        self.subpixel_refinement = True #turn on or off subpixel refinement
        # on for chessboard 
        # off for circular objects
        # set accordingly for custom pattern
        # NOTE: turining on subpixel refinement for circles gives a very high 
        # reprojection error.
        if self.pattern_type in ["asymmetric_circles","symmetric_circles"]:
            self.subpixel_refinement = False
            self.use_clustering = True
            # Setup Default SimpleBlobDetector parameters.
            self.blobParams = cv2.SimpleBlobDetector_Params()
            # Change thresholds
            self.blobParams.minThreshold = 8
            self.blobParams.maxThreshold = 255
            # Filter by Area.
            self.blobParams.filterByArea = True
            self.blobParams.minArea = 50     # minArea may be adjusted to suit for your experiment
            self.blobParams.maxArea = 10e5   # maxArea may be adjusted to suit for your experiment
            # Filter by Circularity
            self.blobParams.filterByCircularity = True
            self.blobParams.minCircularity = 0.8
            # Filter by Convexity
            self.blobParams.filterByConvexity = True
            self.blobParams.minConvexity = 0.87
            # Filter by Inertia
            self.blobParams.filterByInertia = True
            self.blobParams.minInertiaRatio = 0.01
        if self.pattern_type == "asymmetric_circles":
            self.double_count_in_column = True # count the double circles in asymmetrical circular grid along the column
            
        if self.debug_dir and not os.path.isdir(self.debug_dir):
            os.mkdir(self.debug_dir)
                
        print("The Camera Calibration API is initialized and ready for calibration...")
        
    @staticmethod
    def _splitfn(fn):
        path, fn = os.path.split(fn)
        name, ext = os.path.splitext(fn)
        return path, name, ext
        
    
    def _symmetric_world_points(self):
        x,y = np.meshgrid(range(self.pattern_columns),range(self.pattern_rows))
        prod = self.pattern_rows * self.pattern_columns
        pattern_points=np.hstack((x.reshape(prod,1),y.reshape(prod,1),np.zeros((prod,1)))).astype(np.float32)
        return(pattern_points)

    def _asymmetric_world_points(self):
        pattern_points = []
        if self.double_count_in_column:
            for i in range(self.pattern_rows):
                for j in range(self.pattern_columns):
                    x = j/2
                    if j%2 == 0:
                        y = i
                    else:
                        y = i + 0.5
                    pattern_points.append((x,y))
        else:
            for i in range(self.pattern_rows):
                for j in range(self.pattern_columns):
                    y = i/2
                    if i%2 == 0:
                        x = j
                    else:
                        x = j + 0.5
                    
                    pattern_points.append((x,y))
                
        pattern_points = np.hstack((pattern_points,np.zeros((self.pattern_rows*self.pattern_columns,1)))).astype(np.float32)
        return(pattern_points)
        
    def _chessboard_image_points(self,img):
        found, corners = cv2.findChessboardCorners(img,(self.pattern_columns,self.pattern_rows))
        return(found,corners)
    
    def _circulargrid_image_points(self,img,flags,blobDetector):
        found, corners = cv2.findCirclesGrid(img,(self.pattern_columns,self.pattern_rows),
                                             flags=flags,
                                             blobDetector=blobDetector
                                             )
        
        return(found,corners)
        
    def _calc_reprojection_error(self,figure_size=(8,8),save_dir=None):
        """
        Util function to Plot reprojection error
        """
        reprojection_error = []
        for i in range(len(self.calibration_df)):
            imgpoints2, _ = cv2.projectPoints(self.calibration_df.obj_points[i], self.calibration_df.rvecs[i], self.calibration_df.tvecs[i], self.camera_matrix, self.dist_coefs)
            temp_error = cv2.norm(self.calibration_df.img_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            reprojection_error.append(temp_error)
        self.calibration_df['reprojection_error'] = pd.Series(reprojection_error)
        avg_error = np.sum(np.array(reprojection_error))/len(self.calibration_df.obj_points)
        x = [os.path.basename(p) for p in self.calibration_df.image_names]
        y_mean = [avg_error]*len(self.calibration_df.image_names)
        fig,ax = plt.subplots()
        fig.set_figwidth(figure_size[0])
        fig.set_figheight(figure_size[1])
        # Plot the data
        ax.scatter(x,reprojection_error,label='Reprojection error', marker='o') #plot before
        # Plot the average line
        ax.plot(x,y_mean, label='Mean Reprojection error', linestyle='--')
        # Make a legend
        ax.legend(loc='upper right')
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        # name x and y axis
        ax.set_title("Reprojection_error plot")
        ax.set_xlabel("Image_names")
        ax.set_ylabel("Reprojection error in pixels")
        
        if save_dir:
            plt.savefig(os.path.join(save_dir,"reprojection_error.png"))
        
        plt.show()
        print("The Mean Reprojection Error in pixels is:  {}".format(avg_error))
        
    
    def calibrate_camera(self,
                         images_path_list,
                         threads = 4,
                         custom_world_points_function=None,
                         custom_image_points_function=None,
                         ):
        
        """ User facing method to calibrate the camera
        
        Keyword arguments
        
        images_path_list: A list containing full paths to calibration images (No default)
        threads --int: Number of threads to run the calibration (Default 4)
        custom_world_points_function --function: Must be given if pattern_type="custom", else leave at default (Default None)
        custom_image_points_function --function: Must be given if the patter_type="custom", else leave at default (Default None)
        
        A Note on custom_world_points_function() and custom_image_points_function()
        
        * custom_world_points_function(pattern_rows,pattern_columns):
            
        1) This function is responsible for calculating the 3-D world points of the given custom calibration pattern.
        2) Should take in two keyword arguments in the following order: Number of rows in pattern(int), Number of columns in pattern(int)
        3) Must return only a single numpy array of shape (M,3) and type np.float32 or np.float64 with M being the number of control points
           of the custom calibration pattern. The last column of the array (z axis) should be an array of 0
        4) The distance_in_world_units is not multiplied in this case. Hence, account for that inside the function before returning
        5) The world points must be ordered in this specific order : row by row, left to right in every row
        
        * custom_image_points_function(img,pattern_rows,pattern_columns):
            
        1) This function is responsible for finding the 2-D image points from the custom calibration image.
        2) Should take in 3 keyword arguments in the following order: image(numpy array),Number of rows in pattern(int), Number of columns in pattern(int)
        3) This must return 2 variables: return_value, image_points
        4) The first one is a boolean Representing whether all the control points in the calibration images are found
        5) The second one is a numpy array of shape (N,2) of type np.float32 containing the pixel coordinates or the image points of the control points.
           where N is the number of control points.
        6) This function should return True only if all the control points are detected (M = N)
        7) If all the control points are not detected, fillup the 2-D numpy array with 0s entirely and return with bool == False.
        
        
        OUTPUT
        Prints: 
            The calibration log
            plots the reprojection error plot
        
        Returns:
                A dictionary with the follwing keys:
                    return_value of cv2.calibrate_camera --key:'rms'
                    camera intrinsic matrix --key: 'intrinsic_matrix'
                    distortion coeffs --key: 'distortion_coefficients'
                
        Saves:
            Optionally saves the following images if debug directory is specified in the constructor
                                 1.Points visulized on the calibration board
                                 2.Reprojection error plot
        
        """
        
        if self.pattern_type == "custom":
            assert custom_world_points_function is not None, "Must implement a custom_world_points_function for 'custom' pattern "
            assert custom_image_points_function is not None, "Must implement a custom_image_points_function for 'custom' pattern"
            
        # initialize place holders
        img_points = []
        obj_points = []
        working_images = []
        images_path_list.sort()
        print("There are {} {} images given for calibration".format(len(images_path_list),self.pattern_type))
        
        if self.pattern_type == "chessboard":
            pattern_points = self._symmetric_world_points() * self.distance_in_world_units
        
        elif self.pattern_type == "symmetric_circles":
            pattern_points = self._symmetric_world_points() * self.distance_in_world_units
            blobDetector = cv2.SimpleBlobDetector_create(self.blobParams)
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
            if self.use_clustering:
                flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
        
        elif self.pattern_type == "asymmetric_circles":
            pattern_points = self._asymmetric_world_points() * self.distance_in_world_units
            blobDetector = cv2.SimpleBlobDetector_create(self.blobParams)
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            if self.use_clustering:
                flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
                
        elif self.pattern_type == "custom":
            pattern_points = custom_world_points_function(self.pattern_rows,self.pattern_columns)
            
        h, w = cv2.imread(images_path_list[0], 0).shape[:2]
        
        def process_single_image(img_path):
            print("Processing {}".format(img_path))
            img = cv2.imread(img_path,0) # gray scale
            if img is None:
                print("Failed to load {}".format(img_path))
                return None
            
            assert w == img.shape[1] and h == img.shape[0],"All the images must have same shape"
            
            if self.pattern_type == "chessboard":
                found,corners = self._chessboard_image_points(img)
            elif self.pattern_type == "asymmetric_circles" or self.pattern_type == "symmetric_circles":
                found,corners = self._circulargrid_image_points(img,flags,blobDetector)
                
            elif self.pattern_type == "custom":
                found,corners = custom_image_points_function(img,self.pattern_rows,self.pattern_columns)
                assert corners[0] == pattern_points[0], "custom_image_points_function should return a numpy array of length matching the number of control points in the image"
                
            if found:
                #self.working_images.append(img_path)
                if self.subpixel_refinement:    
                    corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), self.term_criteria) 
                else:
                    corners2 = corners.copy()
    
                if self.debug_dir:
                    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    cv2.drawChessboardCorners(vis, (self.pattern_columns,self.pattern_rows), corners2, found)
                    path, name, ext = self._splitfn(img_path)
                    outfile = os.path.join(self.debug_dir, name + '_pts_vis.png')
                    cv2.imwrite(outfile, vis)
                    
            else:
                print("Calibration board NOT FOUND")
                return(None)
            print("Calibration board FOUND")
            return(img_path,corners2,pattern_points)
                
        threads_num = int(threads)
        if threads_num <= 1:
            calibrationBoards = [process_single_image(img_path) for img_path in images_path_list]
        else:
            print("Running with %d threads..." % threads_num)
            pool = ThreadPool(threads_num)
            calibrationBoards = pool.map(process_single_image, images_path_list)
            
        calibrationBoards = [x for x in calibrationBoards if x is not None]
        for (img_path,corners, pattern_points) in calibrationBoards:
            working_images.append(img_path)
            img_points.append(corners)
            obj_points.append(pattern_points)
            
        # combine it to a dataframe
        self.calibration_df = pd.DataFrame({"image_names":working_images,
                                       "img_points":img_points,
                                       "obj_points":obj_points,
                                       })
        self.calibration_df.sort_values("image_names")
        self.calibration_df = self.calibration_df.reset_index(drop=True)
        
        # calibrate the camera
        self.rms, self.camera_matrix, self.dist_coefs, rvecs, tvecs = cv2.calibrateCamera(self.calibration_df.obj_points, self.calibration_df.img_points, (w, h), None, None)
        
        self.calibration_df['rvecs'] = pd.Series(rvecs)
        self.calibration_df['tvecs'] = pd.Series(tvecs)
        
        print("\nRMS:", self.rms)
        print("camera matrix:\n", self.camera_matrix)
        print("distortion coefficients: ", self.dist_coefs.ravel())
        # plot the reprojection error graph
        self._calc_reprojection_error(figure_size=self.figsize,save_dir=self.debug_dir)
        
        result_dictionary = {
                             "rms":self.rms,
                             "intrinsic_matrix":self.camera_matrix,
                             "distortion_coefficients":self.dist_coefs,
                             }
        
        return(result_dictionary)
        
    def visualize_calibration_boards(self,
                                     cam_width = 20.0,
                                     cam_height = 10.0,
                                     scale_focal = 40):
        """
        User facing method to visualize the calibration board orientations in 3-D
        Plots both the pattern centric and the camera centric views
        
        Keyword Arguments: 
        cam_width --float: width of cam in visualization (Default 20.0)
        cam_height --float: height of cam in visualization (Default 10.0)
        scale_focal --int: Focal length is scaled accordingly (Default 40)
        
        Output:
            Plots the camera centric and pattern centric views of the chessboard in 3-D using matplotlib
            Optionally saves these views in the debug directory if the constructor is initialized with 
            debug directory
            
        TIP: change the values of cam_width, cam_height for better visualizations
        """
        
        # Plot the camera centric view
        visualize_views(camera_matrix=self.camera_matrix,
                        rvecs = self.calibration_df.rvecs,
                        tvecs = self.calibration_df.tvecs,
                        board_width=self.pattern_columns,
                        board_height=self.pattern_rows,
                        square_size=self.distance_in_world_units,
                        cam_width = cam_width,
                        cam_height = cam_height,
                        scale_focal = scale_focal,
                        patternCentric = False,
                        figsize = self.figsize,
                        save_dir = self.debug_dir
                        )
        # Plot the pattern centric view
        visualize_views(camera_matrix=self.camera_matrix,
                        rvecs = self.calibration_df.rvecs,
                        tvecs = self.calibration_df.tvecs,
                        board_width=self.pattern_columns,
                        board_height=self.pattern_rows,
                        square_size=self.distance_in_world_units,
                        cam_width = cam_width,
                        cam_height = cam_height,
                        scale_focal = scale_focal,
                        patternCentric = True,
                        figsize = self.figsize,
                        save_dir = self.debug_dir
                        )

#######################################################################################################################
        
## 3-D plotting the pattern centric and camera centric views

def _inverse_homogeneoux_matrix(M):
    # util_function
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv

def _transform_to_matplotlib_frame(cMo, X, inverse=False):
    # util function
    M = np.identity(4)
    M[1,1] = 0
    M[1,2] = 1
    M[2,1] = -1
    M[2,2] = 0

    if inverse:
        return M.dot(_inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))

def _create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
    # util function
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4,5))
    X_img_plane[0:3,0] = [-width, height, f_scale]
    X_img_plane[0:3,1] = [width, height, f_scale]
    X_img_plane[0:3,2] = [width, -height, f_scale]
    X_img_plane[0:3,3] = [-width, -height, f_scale]
    X_img_plane[0:3,4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4,3))
    X_triangle[0:3,0] = [-width, -height, f_scale]
    X_triangle[0:3,1] = [0, -2*height, f_scale]
    X_triangle[0:3,2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4,2))
    X_center1[0:3,0] = [0, 0, 0]
    X_center1[0:3,1] = [-width, height, f_scale]

    X_center2 = np.ones((4,2))
    X_center2[0:3,0] = [0, 0, 0]
    X_center2[0:3,1] = [width, height, f_scale]

    X_center3 = np.ones((4,2))
    X_center3[0:3,0] = [0, 0, 0]
    X_center3[0:3,1] = [width, -height, f_scale]

    X_center4 = np.ones((4,2))
    X_center4[0:3,0] = [0, 0, 0]
    X_center4[0:3,1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4,2))
    X_frame1[0:3,0] = [0, 0, 0]
    X_frame1[0:3,1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4,2))
    X_frame2[0:3,0] = [0, 0, 0]
    X_frame2[0:3,1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4,2))
    X_frame3[0:3,0] = [0, 0, 0]
    X_frame3[0:3,1] = [0, 0, f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]

def _create_board_model(extrinsics, board_width, board_height, square_size, draw_frame_axis=False):
    # util function
    width = board_width*square_size
    height = board_height*square_size

    # draw calibration board
    X_board = np.ones((4,5))
    #X_board_cam = np.ones((extrinsics.shape[0],4,5))
    X_board[0:3,0] = [0,0,0]
    X_board[0:3,1] = [width,0,0]
    X_board[0:3,2] = [width,height,0]
    X_board[0:3,3] = [0,height,0]
    X_board[0:3,4] = [0,0,0]

    # draw board frame axis
    X_frame1 = np.ones((4,2))
    X_frame1[0:3,0] = [0, 0, 0]
    X_frame1[0:3,1] = [height/2, 0, 0]

    X_frame2 = np.ones((4,2))
    X_frame2[0:3,0] = [0, 0, 0]
    X_frame2[0:3,1] = [0, height/2, 0]

    X_frame3 = np.ones((4,2))
    X_frame3[0:3,0] = [0, 0, 0]
    X_frame3[0:3,1] = [0, 0, height/2]

    if draw_frame_axis:
        return [X_board, X_frame1, X_frame2, X_frame3]
    else:
        return [X_board]

def _draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scale_focal,
                       extrinsics, board_width, board_height, square_size,
                       patternCentric):
    # util function
    min_values = np.zeros((3,1))
    min_values = np.inf
    max_values = np.zeros((3,1))
    max_values = -np.inf

    if patternCentric:
        X_moving = _create_camera_model(camera_matrix, cam_width, cam_height, scale_focal)
        X_static = _create_board_model(extrinsics, board_width, board_height, square_size)
    else:
        X_static = _create_camera_model(camera_matrix, cam_width, cam_height, scale_focal, True)
        X_moving = _create_board_model(extrinsics, board_width, board_height, square_size)

    cm_subsection = linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [ cm.jet(x) for x in cm_subsection ]

    for i in range(len(X_static)):
        X = np.zeros(X_static[i].shape)
        for j in range(X_static[i].shape[1]):
            X[:,j] = _transform_to_matplotlib_frame(np.eye(4), X_static[i][:,j])
        ax.plot3D(X[0,:], X[1,:], X[2,:], color='r')
        min_values = np.minimum(min_values, X[0:3,:].min(1))
        max_values = np.maximum(max_values, X[0:3,:].max(1))

    for idx in range(extrinsics.shape[0]):
        R, _ = cv2.Rodrigues(extrinsics[idx,0:3])
        cMo = np.eye(4,4)
        cMo[0:3,0:3] = R
        cMo[0:3,3] = extrinsics[idx,3:6]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4,j] = _transform_to_matplotlib_frame(cMo, X_moving[i][0:4,j], patternCentric)
            ax.plot3D(X[0,:], X[1,:], X[2,:], color=colors[idx])
            min_values = np.minimum(min_values, X[0:3,:].min(1))
            max_values = np.maximum(max_values, X[0:3,:].max(1))

    return min_values, max_values

def visualize_views(camera_matrix,
                      rvecs,
                      tvecs,
                      board_width,
                      board_height,
                      square_size,
                      cam_width = 64/2,
                      cam_height = 48/2,
                      scale_focal = 40,
                      patternCentric = False,
                      figsize = (8,8),
                      save_dir = None
                          ):
    """
    Visualizes the pattern centric or the camera centric views of chess board
    using the above util functions
    
    Keyword Arguments
    
    camera_matrix --numpy.array: intrinsic camera matrix (No default)
    rvecs : --list of rvecs from cv2.calibrateCamera()
    tvecs : --list of tvecs from cv2.calibrateCamera()
    
    board_width --int: the chessboard width (no default)
    board_height --int: the chessboard height (no default)
    square_size --int: the square size of each chessboard square in mm
    cam_width --float: Width/2 of the displayed camera (Default 64/2)
                       it is recommended to leave this argument to default
    cam_height --float: Height/2 of the displayed camera (Default (48/2))
                        it is recommended to leave this argument to default
    scale_focal --int: Value to scale the focal length (Default 40)
                       it is recommended to leave this argument to default
    
    pattern_centric --bool: Whether to visualize the pattern centric or the
                            camera centric (Default False)
    fig_size --tuple: The size of figure to display (Default (8,8))
                      it is recommended to leave this argument to default
    
    save_dir --str: optional path to a saving directory to save the 
                    generated plot (Default None)
                    
    Does not return anything
    """
    i = 0
    extrinsics = np.zeros((len(rvecs),6))
    for rot,trans in zip(rvecs,tvecs):
        extrinsics[i]=np.append(rot.flatten(),trans.flatten())
        i+=1
    #The extrinsics  matrix is of shape (N,6) (No default)
    #Where N is the number of board patterns
    #the first 3  columns are rotational vectors
    #the last 3 columns are translational vectors
    
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")

    min_values, max_values = _draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, patternCentric)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    if patternCentric:
        ax.set_title('Pattern Centric View')
        if save_dir:
            plt.savefig(os.path.join(save_dir,"pattern_centric_view.png"))
    else:
        ax.set_title('Camera Centric View')
        if save_dir:
            plt.savefig(os.path.join(save_dir,"camera_centric_view.png"))
    plt.show()   
#################################################################################################################
    
if __name__ == "__main__":
    ## Cannot be used for custom calibration pattern
    parser = argparse.ArgumentParser(description="Camera_Calibration_API. Saves the calibration results in a pickle file \n NOTE: USE THE API AS IMPORTABLE MODULE FOR ADDED CONTROL",formatter_class=RawTextHelpFormatter)
    parser.add_argument("--images_dir",help="Path to the directory containing calibration images (no / in end)",type=str,metavar='', default=None)
    parser.add_argument("-pt","--pattern_type",help="The pattern type for calibration",type=str,metavar='',default=None)
    parser.add_argument("-pr","--pattern_rows",help="num of rows in pattern",type=int,metavar='',default = 0)
    parser.add_argument("-pc","--pattern_columns",help="num of columns in pattern",type=int,metavar='',default = 0)
    parser.add_argument("-d","--distance",help="The distance between points in world units",type=float,metavar='',default = 1.0)
    parser.add_argument("--debug",help="path to directory for saving images",type=str,metavar='',default=None)
    parser.add_argument("-cw","--cam_width",help="width of cam for visualization",type=float,metavar='',default=1)
    parser.add_argument("-ch","--cam_height",help="height of cam for visualization",type=float,metavar='',default=0.5)
    parser.add_argument("--save",help="path to save the results as a pickle file",type=str,metavar='',default="./results.pickle")
    
    args=parser.parse_args()
    
    pattern_types = ["chessboard","symmetric_circles","asymmetric_circles"]
    
    if args.images_dir == None or args.pattern_type == None or args.pattern_rows == 0 or args.pattern_columns == 0:
        raise ValueError("Give values for the first 4 arguments")
    assert args.pattern_type in pattern_types, "The --pattern_type must be one of {}. 'custom' pattern is not supported in terminal mode".format(pattern_types)
        
    file_types = ("*.jpg","*.jpeg","*.JPEG","*.png","*.PNG","*.bmp","*.BMP")
    images_path_list = []
    for file_type in file_types:
        images_path_list.extend(glob.glob(os.path.join(args.images_dir,file_type)))
    
    calibration_object = Camera_Calibration_API(pattern_type = args.pattern_type,
                                                pattern_rows = args.pattern_rows,
                                                pattern_columns = args.pattern_columns,
                                                distance_in_world_units = args.distance,
                                                debug_dir = args.debug
                                                )
    
    results = calibration_object.calibrate_camera(images_path_list)
    with open(args.save,"wb") as f:
        pickle.dump(results,f)
    
    calibration_object.visualize_calibration_boards(cam_width=args.cam_width,
                                                    cam_height=args.cam_height)
    
    
    
        

   
    