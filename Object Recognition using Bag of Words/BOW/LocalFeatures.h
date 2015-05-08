/**
 * Author: Manan Vyas
 * USCID:  7483-8632-00
 * Email:  mvyas@usc.edu
 */


// C/C++ specific
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>
#include <fstream>
#include <vector>

// OpenCV specific
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/legacy/legacy.hpp>

#ifndef __LOCALFEATURES__
#define __LOCALFEATURES__

// Mechanism to read all the training/Testing datasets
void myIO(int kNumofTrainingImagesSet);

// Compute SIFT features
cv::Mat SIFT_features(cv::Mat img, int retVal);

// PCA
cv::Mat principalComponentAnalysis (cv::Mat SIFT_features);

// Generate CodeWords for PCA-SIFT
cv::Mat generateCodewords (cv::Mat eigenVectors, cv::Mat SIFT_features);

// Kmeans
cv::Mat K_Means (cv::Mat codeWords, int kClusters);

// Object Feature Vector


#endif
