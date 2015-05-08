/* ****
 *  CSCI 574 Computer Vision Homework 5
 *  Object recognition and Matching using Bag od Words classifier
 *  Name : Manan Vyas
 *  USC ID : 7483-8632-00
 *  Email : mvyas@usc.edu
 ****/

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	// get PCA-SIFT featrures
	Mat siftFeatures;
	int siftNum[100];
	int ct=0;
	string name;


	int const kComponents = 30 ;
	int const kClusters = 100 ;
	int const kNearestNeighbors = 10;

	while (ct!=5)
	{
		switch(ct)
		{
		case 0:
			name = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/car/";
			break;
		case 1:
			name = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/cougar/";
			break;
		case 2:
			name = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/face/";
			break;
		case 3:
			name = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/pizza/";
			break;
		case 4:
			name = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/training/sunflower/";
			break;
		}

		for (int i=1; i<=20; i++)
		{
			ostringstream oss;
			string file;

			if(i < 10)
			{
				oss << i;
				file = name + "image_000" + oss.str() + ".jpg";
			}
			else
			{
				oss << i;
				file = name + "image_00" + oss.str() + ".jpg";
			}

			Mat grayimg;
			if(ct!=0)
			{
				Mat img = imread(file, 1);
				cvtColor(img, grayimg, CV_BGR2GRAY);
			}
			else
			{
				grayimg = imread(file, 0);
			}

			SIFT sift;
			vector<KeyPoint> keypoints;
			Mat descriptors;
			sift.operator()(grayimg, noArray(), keypoints, descriptors, false);
			siftFeatures.push_back(descriptors);
			siftNum[ct*20 + i - 1] = descriptors.rows;
		}
		ct++;
	}



	Mat covar, mean;
	//const Mat *samples = &siftFeatures;
	calcCovarMatrix(siftFeatures, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_64F);
	Mat eigenvalues, eigenvectors;
	eigen(covar, eigenvalues, eigenvectors);
	Mat priDir;



	for(int i=0; i<(kComponents); i++) //PCA components
		priDir.push_back(eigenvectors.row(i));

	Mat priDir_t = priDir.t();
	Mat sift64f;
	siftFeatures.convertTo(sift64f, CV_64F);
	Mat pcaSift = sift64f * priDir_t;


	// k-means clustering
	Mat bestLabels;
	Mat pcaSift32f;
	pcaSift.convertTo(pcaSift32f, CV_32F);
	Mat centers;
	//const int K = 100; // clusters //10 attempts
	kmeans(pcaSift32f, (kClusters), bestLabels, TermCriteria( CV_TERMCRIT_EPS, 100, 0.55), 10, KMEANS_PP_CENTERS, centers);



	// compute histogram
	int sum[100][kClusters] = {0};
	int step = 0;
	for(int i=0; i<100; i++)
	{
		for(int j=0; j<siftNum[i]; j++)
		{
			if(i==0)
				sum[i][bestLabels.at<int>(j)]++;
			else
				sum[i][bestLabels.at<int>(j + step)]++;
			//cout<<bestLabels.at<int>(j);
		}
		step = step + siftNum[i];
	}

    // Print training histogram
/*
	ofstream myfile;
	myfile.open ("histogram.csv");
	//printMat(&sum);
	//myfile<<"P = "<<endl;
	for(int i=0;i<100;i++)
	{

		for(int k=0;k<(kClusters);k++)
			myfile<< sum[i][k]<<",";
		myfile<<endl;
	}
	myfile.close();
*/

	// Object  Recognition
	for (int ii=21; ii<=30; ii++)
	{
		ostringstream oss;
		string file;

        oss << ii;
        file = "M:/Study-Fall14/CSCI574/Homeworks/HW5/dataset/images/testing/sunflower/image_00" + oss.str() + ".jpg"; //

		//Mat testImg = imread("test_car_10.jpg", 0);
		Mat testImg = imread(file, 1);
		Mat grayTestImg;
		cvtColor(testImg, grayTestImg, CV_BGR2GRAY);
		SIFT sift;
		vector<KeyPoint> keypoints;
		Mat descriptors;
		sift.operator()(grayTestImg, noArray(), keypoints, descriptors, false);
		Mat feature;
		Mat descriptors64f;
		descriptors.convertTo(descriptors64f, CV_64F);
		feature = descriptors64f * priDir_t;

		double **distance;  // i: each feature; j: to each center
		distance = new double * [feature.rows];
		for(int i=0; i<feature.rows; i++)
        {
			distance[i] = new double [kClusters];
			for (int j=0; j< (kClusters); j++)
				distance[i][j] = 0;
		}

		for(int i=0; i<feature.rows; i++)
        {
            for(int j=0; j<((kClusters)); j++) // K is the number of cluster centers
            {
				for(int k=0; k<(kComponents); k++) //
                {
					distance[i][j] += pow(feature.at<double>(i,k) - centers.at<float>(j,k), 2.0);
                }
				distance[i][j] = sqrt(distance[i][j]);
			}
        }

		int histo[kClusters] = {0};
		vector<int> minDistances;
		for(int i=0; i<feature.rows; i++)
		{
			int temp=0;
			for(int j=1; j<((kClusters)); j++)
			{
				if(distance[i][temp] > distance[i][j])
					temp = j;
			}
			minDistances.push_back(temp);
		}

		for(int i=0; i<feature.rows; i++)
		{
			histo[minDistances[i]]++;
		}

		/*for(int i=0; i<feature.rows; i++)
		{
			double minDistances = distance[i][0];
			int temp=0;
			for(int j=0; j<K; j++)
			{
				if(minDistances > distance[i][j]) {
					minDistances = distance[i][j];
					temp = j;
				}
			}
			histo[temp] ++;
		}*/



		double diff[100] = {0};
		for(int i=0; i<100; i++)
        {
			for(int j=0; j<((kClusters)); j++)
			{
				diff[i] += (histo[j] - sum[i][j]) * (histo[j] - sum[i][j]);
			}
			diff[i] = sqrt(diff[i]);
		}

		const int neib = 10; // number of neighbors
		Mat dst;
		Mat src(1,100,CV_64F,&diff);
		sortIdx(src,dst,CV_SORT_EVERY_ROW | CV_SORT_ASCENDING );
        
        // Vote Based KNN

		double vote[5] = {0};
		for(int i=0; i<(kNearestNeighbors); i++)
		{
			if(dst.at<int>(i) >= 0 && dst.at<int>(i) < 20)
				vote[0] += 1/diff[dst.at<int>(i)];
			else if(dst.at<int>(i) >= 20 && dst.at<int>(i) < 40)
				vote[1] += 1/diff[dst.at<int>(i)];
			else if(dst.at<int>(i) >= 40 && dst.at<int>(i) < 60)
				vote[2] += 1/diff[dst.at<int>(i)];
			else if(dst.at<int>(i) >= 60 && dst.at<int>(i) < 80)
				vote[3] += 1/diff[dst.at<int>(i)];
			else
				vote[4] += 1/diff[dst.at<int>(i)];

			//cout<<dst.at<int>(i)<<"    "<<diff[dst.at<int>(i)]<<endl;
		}
		int max=0;
		for(int i=0; i<5; i++)
		{
			if(vote[max] < vote[i])
				max=i;
		}
		switch(max)
        {
			case 0:
				cout<<"Image_00"<<ii<<" is: car"<<endl;
				break;
			case 1:
				cout<<"Image_00"<<ii<<" is: cougar"<<endl;
				break;
			case 2:
				cout<<"Image_00"<<ii<<" is: face"<<endl;
				break;
			case 3:
				cout<<"Image_00"<<ii<<" is: pizza"<<endl;
				break;
			case 4:
				cout<<"Image_00"<<ii<<" is: sunflower"<<endl;
				break;
        }
		delete[] distance;
	}
	system("pause");
	return 0;
}
