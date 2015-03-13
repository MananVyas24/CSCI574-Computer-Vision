/* *******
    Name : Manan Vijay Vyas
    USCID: xxxxxxxxxx
    Email: mvyas@usc.edu

    CSCI 574 : Homework#4

******** */

#include <opencv/cv.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

void process_ ()
{
    Mat U, Vt, D, A, P, temp, temp1, D1_sqrt;

    int const kNumberOfImages = 6 ;
    int const kNumberOfPoints = 8;

    // Input Co-Ordinates for homework # 4
    Mat W = (Mat_<double>(2*kNumberOfImages,kNumberOfPoints) << 130, 150, 195, 172, 111, 130, 174, 152,
         196, 230, 159, 125, 196, 231, 158, 123,
         86, 64, 110, 134, 100, 78, 123, 147,
         119, 113, 47, 50, 131, 126, 60, 63,
         15, 38, 76, 54, 27, 49, 89, 67,
         129, 129, 72, 70, 142, 141, 84, 83,
         27, 63, 74, 35, 21, 58, 67, 27,
         18, 31, 40, 26, 36, 49, 60, 46,
         48, 72, 135, 113, 43, 67, 130,
         108, 53, 55, 76, 75, 72, 74, 96,
         95, 130, 145, 46, 37, 131, 145, 48,
         38, 29, 29, 32, 32, 53, 54, 58, 57);

    // Average matrix
    Mat W_ (Size(1,12),CV_64F);
    for (unsigned int i = 0 ; i < (2*kNumberOfImages) ; i++)
    {
        Scalar sumX_Y = cv::sum(W.row(i));
        W_.at<double>(i, 0) = ((double)sumX_Y[0]) / ((double)kNumberOfPoints);
    }

    // Compute distance for average for each co-ordinate to get the actual W matrix for SVD
    for (unsigned int i = 0 ; i < (2*kNumberOfImages) ; i++)
    {
		W.row(i) -= Mat::ones(Size(kNumberOfPoints, 1), CV_64F) * (W_.at<double>(i, 0));
	}

    // Compute SVD
    SVD::compute(W, D, U, Vt);

    // Algorithm 6.2 from text book
    Rect roi1(0, 0, 3, 12);
    Rect roi2(0, 0, 3, 8);
    Mat U1(U, roi1);
    transpose(Vt, temp);
    Mat Vt1(temp, roi2);
    Mat D1 = (Mat_<double>(3,3) << D.at<double>(0,0), 0, 0, 0,
    D.at<double>(1,0), 0, 0, 0, D.at<double>(2,0));
    // the Sqrt transpose method to compute P
    sqrt(D1, D1_sqrt);
    transpose(Vt1, temp1);
    A = U1 * D1_sqrt;

    P = D1_sqrt * temp1;

    cout << "W = " << endl << W << endl;
    cout << "U1 = " << endl << U1 << endl;
    cout << "D1_sqrt = " << endl << D1_sqrt << endl;
    cout << "V_transpose = " << endl << temp1 << endl;
    //cout << "D = " << endl << W << endl;
    cout << "P = " << endl << P << endl;

}

int main()
{
    process_ ();
    //waitkey(0);
    return 0;
}

/*
P =
[10.68738697719022, 11.95551382929746, 10.77165019507828, 9.498434834764916, 11.
08447847900898, 12.34859144897835, 11.14330366828817, 9.860168648547067;
  -5.303208060341911, -5.001824783800161, 6.184852695940656, 5.831138723870756,
-5.543576843486757, -5.263975143306675, 5.901585268985403, 5.593843579755961;
  1.160920991146074, -4.472669793817366, -4.408059110869711, 1.373532631986038,
4.537978694365356, -1.102153197087567, -0.8951262596487105, 4.947693301752587]
*/
