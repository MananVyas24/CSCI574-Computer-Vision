
#ifndef ASSIGNMENT2_H_
#define ASSIGNMENT2_H_

// include necessary header files
#include <opencv/cv.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#define PIXEL_THRESH 225
#define PIXEL_THRESH_NEG -225
#define PIXEL_DIST 2

// define namespaces
using namespace cv;
using namespace std;


class Assignment2
{
private:
	Mat image;
	vector<cv::KeyPoint> keypoints;
	Mat descriptors;
	Mat sift_output;
	vector< DMatch > matches;

public:
	Assignment2();

	virtual ~Assignment2();

	Assignment2(Mat m);

	void extractSiftFeatures();

	vector< DMatch > FindMatchesEuclidian(Assignment2 &m2);

	void displayFeatures(Assignment2 &m2);

	void displayGoodMatches(Assignment2 &m2, const vector< DMatch> &good_matches);

	Mat computeRANSAC_opencv(Assignment2 &m2);

	Mat computeRANSAC(Assignment2 &m2);

	vector<int> randomGenerate(int N, int M);

	Mat computeH(const vector<Point2f> &query, const vector<Point2f> &train);

	int computeInliers(const vector<Point2f> &train, const vector<Point2f> &query, Mat &H, double thresh);

	void normalizeH(Mat Hbest);

	Mat warpImage(Mat H);

	Mat displayOverlaying(Assignment2 &m2, Mat H, Mat image1);

	inline Mat getSIFTImage() { return sift_output; }

};

#endif /* ASSIGNMENT2_H_ */
