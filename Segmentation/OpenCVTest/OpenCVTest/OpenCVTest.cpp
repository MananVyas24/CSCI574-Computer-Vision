/* ------------------------------------------------------------------------
Name: Manan Vyas
USCID: 7483-8632-00
Email: mvyas@usc.edu
CSCI 574 HW#2: Implementation of Mean Shift Segmentation and Watershed Segmentation
---------------------------------------------------------------------------*/
#include "stdafx.h"
// OpenCV Specific Includes ...
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
// General imports/includes
#include <iostream>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
// Functionaly to generate seed pixels and then extract intensity info using Watershed to generate to output intensities
#include "SeedPixel.h"
// Namespaces
using namespace cv;
using namespace std;

// Initialize a Vector for seedpixel as markers
vector<SeedPixel> SeedPixels;

// Fucntion to Display the Image in a Seperate Window rendered on the screen ...
void DisplayImage (Mat img,String WindowName)
{
	namedWindow (WindowName,WINDOW_AUTOSIZE);
	imshow (WindowName,img);
}

// RGB -> Other Color Space conversion conversions ..
Mat ColorSpaceConversion (Mat img,const string ColorSpace)
{
	// RGB to CIE L*a*b color space ...
	Mat imgLAB ; // output img in LAB color format ...
	cvtColor(img,imgLAB,COLOR_BGR2Lab);
	return imgLAB;
}

// Mean Shift Segmentation
Mat MeanShiftSegmentation (Mat img,int SpatialRadius,int IntensityRadius,int PyramidLevels) 
// Here we will use pyramid levels = 1 as asked in the problem
{
	Mat meanshiftsegmentedImage;
	// Actual Mean Shifting Segmentation...
	// Invoke the Mean Shift Segmented image built in of OpenCV
	// Reference : URL <http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=pyrmeanshiftfiltering#cv.PyrMeanShiftFiltering>
	pyrMeanShiftFiltering (img,meanshiftsegmentedImage,SpatialRadius,IntensityRadius,PyramidLevels);
	// return the output image ...
	return meanshiftsegmentedImage;
}

/* Method to perform watershed segmentation
 * Reference : URL <http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=watershed#void watershed(InputArray image, InputOutputArray markers)>
 */
Mat WatershedSegmentation (Mat img,int kHeight,int kWidth)
{
	Mat markerMask,imgGray;
	// Initialize the total number of comonents that acts as seeds within the image
	int componentCount = 0;
	// size factor for seeds ... explained in detail while generating the seed pixels below..
	int seed_size = 25;

	// Generate the Marker Map
	cvtColor(img, markerMask, CV_BGR2GRAY);
	// Marker Mask
	cvtColor(markerMask,imgGray,CV_GRAY2BGR);
	Mat markers(markerMask.size(),CV_32S); 
	markers = Scalar::all(0);

	// Reserve the vector capacity
	// Reference : URL <http://www.cplusplus.com/reference/vector/vector/reserve/>
	SeedPixels.reserve(componentCount);

	// Initializing seeds as a sort of 'SeedPixels' that are 
	// are circular regions of unit radius at the uniform distance of 'seed_size'
	// away from each other at along the rows as well as the columns.

	for (int i=0;i<kHeight;i++)
	{
		for (int j=0;j<kWidth;j++)
		{
			if(i % seed_size == 0 && j % seed_size == 0)
			{	
				// Reference : URL <http://docs.opencv.org/modules/core/doc/drawing_functions.html>
				// Circle drawing function : 
				// C++: void circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
				circle(markers,Point(j,i),1,Scalar::all(componentCount+1),-1,20,0);
				SeedPixel sp = SeedPixel();
				SeedPixels.push_back(sp);
				// Count the total number of components that act as seed pixels
				componentCount++; 
			}
			//Contour Methods ... ?
		}
	}
	DisplayImage(markers,"WaterShed Markers");
	// Run Watershed ... by invoking the OpenCV in built function
	watershed(img,markers);
	// Display WaterShed with markers ...
	//DisplayImage(markers,"WaterShed Markers");
	Mat SegmentedImage(markers.size(), CV_8UC3);
	//cout<<"Marker Size : "<<markers.size()<<endl;
	//cout<<"Comp Count : "<<componentCount<<endl;

	// paint the watershed image with the seed intensities
	// Reference : OpenCV Watershed demo example
	// URL: <https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/cpp/watershed.cpp?rev=4270>
	// Dated 09/28/14
	for(int i = 0; i < markers.rows; i++) 
	{
		for(int j = 0; j < markers.cols; j++)
		{
			// Retrieve the index of the markers at position i,j within the image
			// and then initialize the pixels within the segmented image
			int index = markers.at<int>(i, j);
			// Check the conditions on the markers ...
			if(index == -1)
				SegmentedImage.at<Vec3b>(i, j) = img.at<Vec3b>(i,j);
			else if(index <= 0 || index > componentCount)
				SegmentedImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			else
			{  
				SeedPixels.at(index-1).feedPixel(Point(i,j));
			}
		}
	}

	// Once we have the total number of components with us .. 
	// paint the output image with the representative intensities
	// that were calculated by 'calculateColorofSeedRegion' method
	// and then return the Segmented Image as the output ...
	for(int i = 0; i < componentCount; i++)
	{
		SeedPixels.at(i).calculateColorofSeedRegion(img);
	}
	for(int i = 0; i < markers.rows; i++)
	{
		for(int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if(index > 0 && index < componentCount)
				SegmentedImage.at<Vec3b>(i,j) = SeedPixels.at(index-1).m_color;
		}
	}
	return SegmentedImage;
}

int main(int argc, char* argv[])
{	
	
	// Read the input Image as the the program argument given by the user 
	// Input Image has 24bit depth ... 8-bit per channel BGR values ...
	Mat img = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	/*
	Mat img1 = imread(argv[2],CV_LOAD_IMAGE_COLOR);
	Mat img2 = imread(argv[3],CV_LOAD_IMAGE_COLOR);
	Mat img3 = imread(argv[4],CV_LOAD_IMAGE_COLOR);
	*/
	// Format being pixel = | 8bit Blue channel | 8bit Green Channel | 8bit Red Channel|
	//DisplayImage(img,"InputImage");
	// Check for Validity of Image ...
	/*
	if(!img1.data || !img2.data || !img3.data )                             
    {
        cout <<  "Could not open or find the image" <<endl ;
        return -1;
    }*/

	// print RGB values ... # of rows = height ; # of cols = width
	unsigned char *input = (unsigned char*)(img.data); //Image data
	const int kHeight = img.rows;
	const int kWidth = img.cols;
	const int kBytesPerPixel = img.channels();
	// Spatial and Intensity radii for the mean shift segmentation ...
	const int kSpatialRadius = 2;
	const int kIntensityRadius = 2;
	/*
	//-----------------------  Mean shift segmentation --------------------------
	Mat imgLAB; Mat MSImg; 
	// Convert the image from RGB to CIE LAB
	cvtColor(img,imgLAB,COLOR_BGR2Lab);
	// Display the L*a*b image
	DisplayImage (imgLAB,"Input Image in LAB Color Space");
	// Segment the Image using Mean Shift in L*a*b space
	Mat outputImg = MeanShiftSegmentation(imgLAB,kSpatialRadius,kIntensityRadius,1);
	// Convert back the Image from L*a*b to RGB image and then display
	cvtColor(outputImg,MSImg,COLOR_Lab2BGR);
	// Display Image ...
	DisplayImage (MSImg,"Mean Shift Segmented Image");
	*/

	//----------------------- Watershed Segmentation ---------------------------
	Mat wshedImg1 = WatershedSegmentation(img,kHeight,kWidth);
	/*
	Mat wshedImg2 = WatershedSegmentation(img1,kHeight,kWidth);
	Mat wshedImg3 = WatershedSegmentation(img2,kHeight,kWidth);
	Mat wshedImg4 = WatershedSegmentation(img3,kHeight,kWidth);*/

	DisplayImage (wshedImg1,"Watershed Transformed Image1");
	/*
	DisplayImage (wshedImg2,"Watershed Transformed Image2");
	DisplayImage (wshedImg3,"Watershed Transformed Image3");
	DisplayImage (wshedImg4,"Watershed Transformed Image4");*/

	// ------------------ End of program --------------------------------------	
	waitKey(0);
	// Wait for the user to press key on the current window and then exit the program
    return 0;
}

