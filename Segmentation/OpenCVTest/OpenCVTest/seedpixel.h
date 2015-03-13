/* Name : Manan Vyas
 * USCID: 7483-8632-00
 * Email : mvyas@usc.edu 
 * CSCI 574 : HW#2 Mean Shift Segementation and Watershed Segmentation.
 */

// Defines the class to handle the seed pixels for the watershed segmentation
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class SeedPixel
{
	private:
		vector<Point> m_pixels;
		int m_total;
	public:
		SeedPixel();
		void feedPixel(Point pos);
		void calculateColorofSeedRegion(Mat& InputImage);
		Vec3b m_color;
};

/*------------------------
 * Constructor...
 ------------------------*/
SeedPixel::SeedPixel(){}


/*
 * Feed in a pixel from the reference image
 * and push back its poition into the defined vector.
 */
void SeedPixel::feedPixel(Point pos)
{
	m_pixels.push_back(pos);
}

/*
 * Calculate the color of the "SeedPixel" circular region...
 * which are the seed pixels that marks for the watershed segmntation
 */
void SeedPixel::calculateColorofSeedRegion(Mat& InputImage)
{	
	// total number of pixels present within the cicular region of 'SeedPixel'
	int R=0, G=0, B=0;
	m_total = m_pixels.size();
	
	//cout<<"m_total: "<<m_total<<endl;

	// Traverse through the circular 'SeedPixel' region
	for(int i = 0; i < m_total; i++)
	{
		// get the current position of the pixel [x,y] within the image...
		int x = m_pixels.at(i).x;
		int y = m_pixels.at(i).y;
		// Their intensity vector...
		Vec3b intensity = InputImage.at<Vec3b>(x, y);

		// Accumulate intensities of all the color channels to
		// get a representative intensity of the 'SeedPixel' circular region ...
		R += (int)InputImage.at<Vec3b>(x, y).val[0];
		G += (int)InputImage.at<Vec3b>(x, y).val[1];
		B += (int)InputImage.at<Vec3b>(x, y).val[2];
	}
	// These are the normalized RGB values that represent the [RGB] values of each pixel present 
	// within that particluar SeedPixel ...
	R = ((R==0) ? 0 : R/m_total);
	G = ((G==0) ? 0 : G/m_total);
	B = ((B==0) ? 0 : B/m_total);

	// set the color channel RGB which will then used to write into the output image...
	m_color = Vec3b((uchar)(R), (uchar)(G), (uchar)(B));
}
