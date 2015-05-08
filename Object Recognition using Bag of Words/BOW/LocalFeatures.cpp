/**
 * Author: Manan Vyas
 * USCID:  7483-8632-00
 * Email:  mvyas@usc.edu
 */

// header
#include "LocalFeatures.h"

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

using namespace std;
using namespace cv;

//***
// Method to calculate SIFT_PCA features
//***
Mat SIFT_features(Mat img,int retVal)
{
    SIFT siftFeatureCompute;
    vector<KeyPoint> SIFTkeypoints;
    Mat SIFT_features;
    Mat descriptor;
    siftFeatureCompute.operator()(img, noArray(), SIFTkeypoints, descriptor, false);
    SIFT_features.push_back(descriptor);

    if (retVal == 1)
        return descriptor;
    else
        return SIFT_features;
}

// ***
// Method to perform PCA using the eigen vectors and reduce the dimensionaltiy of the problem
// ***
Mat principalComponentAnalysis (Mat SIFT_features)
{
    Mat covarianceMatrix, mean;
    Mat eigenValues,eigenVectors;

    // Calculate covariance matrix for PCA
    // Reference : OpenCV docs
    //URL<http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?
    //    highlight=calccovarmatrix#void%20calcCovarMatrix
    //    (const%20Mat*%20samples,%20int%20nsamples,%20Mat&%20covar,%20Mat&%20mean,%20int%20flags,%20int%20ctype)>
    calcCovarMatrix(SIFT_features, covarianceMatrix, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_64F);

    // Calculate Eigen decomposition to reduce the dimensionalty
    // Reference OpenCV docs :
    // <URL><http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?
    // highlight=eigen#bool%20eigen
    // (InputArray%20src,%20OutputArray%20eigenvalues,%20OutputArray%20eigenvectors,%20int%20lowindex,int%20highindex)>
    eigen(covarianceMatrix, eigenValues, eigenVectors);

    return eigenVectors;
}

// ***
//Funtion to generate PCA-SIFT codewords like in VQ
// ***
Mat generateCodewords (Mat eigenVectors, Mat SIFT_features)
{
    Mat principalDirections,pDt;
    Mat codeWords;
    // 20 components considered as advised in the homeword question ...
    for(unsigned int i=0; i<20; i++)
        principalDirections.push_back(eigenVectors.row(i));

    // .T !
    pDt = principalDirections.t();
    Mat temp;
    (SIFT_features.convertTo(temp, CV_64F));
    // Generate codeWords...
    codeWords = (temp)*(pDt);

    return codeWords;
}

// ***
// K-Means Clustering on PCA-SIFT codeWords genetared
// kClsters around 100-120 range is good
// ***
Mat K_Means (Mat codeWords, int kClusters)
{
    Mat labels, centers;
    codeWords.convertTo(codeWords, CV_32F);

    // K-Means .. with 10 attempts
    kmeans(codeWords, kClusters, labels, TermCriteria(CV_TERMCRIT_EPS, 100, 0.48), 10, KMEANS_PP_CENTERS, centers );

    return labels;
}

// ***
// Object Feature vector generation method
// Generates a histogram of codewords for each image to obtain its unique feature vector
// ***
void featureVector ()
{

}

// ***
// Function to read and analyse the input images ...
// This calculates and analysis the input images WRT training data
// ***
void myIO(int kNumofTrainingImagesSet)
{
    int countVar = 0;
    string trainingSetImageName;

    Mat descriptors;

    // Points of interest for training the system/recognizer
    int trainingPoints[100] = {0};

    while (countVar != kNumofTrainingImagesSet)
    {
        // Switch case to select the appropriate folder structure
        switch(countVar)
        {
            case 0 :
                trainingSetImageName = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/car/";
                break;
            case 1 :
                trainingSetImageName = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/cougar/";
                break;
            case 2 :
                trainingSetImageName = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/face/";
                break;
            case 3 :
                trainingSetImageName = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/pizza/";
                break;
            case 4 :
                trainingSetImageName = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/sunflower/";
                break;
        }
        // Read each image set within the training set
        for (unsigned int i=1 ; i<=20 ; i++)
        {
            Mat img; // To store actual individual images that areour training data
            //stringstream
            ostringstream oss;
            // final individual image filename
            string fileName;
            // 0001->0020 are the image names
            if (i<10) // 0001->0009
            {
                oss << i;
                fileName = trainingSetImageName+"image_000"+oss.str()+".jpg";
            }
            else // 0010->0020
            {
                oss << i;
                fileName = "image_00"+oss.str()+".jpg";
            }
            //*completed read all 20 images from one dataset of images
            // Read indiavial images into Mats
            // Note : Car images are grey images in the given data set .. hence the if..else structure .. not really needed

            if (countVar == 0) // Read grey Images
            {
                img = imread(fileName,0);

            }
            else // color image
            {
                cvtColor((imread(fileName,1)),img,COLOR_BGR2GRAY);
            }

            // Calculate SIFT_PCA Features and store it in the data structure ...
            descriptors = SIFT_features(img, 1);
            trainingPoints[countVar*20 + i - 1] = descriptors.rows;

        }// end of for loop
        // next 'SET' of training Images
        countVar = countVar+1;
    }//end of loop
}//end of myIO function ...

