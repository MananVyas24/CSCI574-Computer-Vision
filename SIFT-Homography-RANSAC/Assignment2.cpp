
#include "Assignment2.h"

Assignment2::Assignment2()
{
	// TODO Auto-generated constructor stub
}

Assignment2::~Assignment2()
{
	// TODO Auto-generated destructor stub
}

Assignment2::Assignment2(cv::Mat m)
{
	image = m.clone();
}

void Assignment2::extractSiftFeatures()
{
	SIFT siftobject;
	siftobject.operator()(image, Mat(), keypoints, descriptors); // Apply SIFT operator
}

vector< DMatch > Assignment2::FindMatchesEuclidian(Assignment2 &m2)
{
	//-----------------------------------------------------------------------------------------------------------------//
	//--- Calculate euclidian distance between keypoints to find best matching pairs.
	// crete two dimensional vector for storing euclidian distance
	vector< vector<float> > vec1, unsortedvec1;
	for (int i=0; i<this->keypoints.size(); i++)
	{
		vec1.push_back(vector<float>()); // Add an empty row
		unsortedvec1.push_back(vector<float>());
	}

	// create vector of DMatch for storing matxhes point
	vector< DMatch > matches1;
	DMatch dm1;

	// loop through keypoints1.size
	for (int i=0; i<this->keypoints.size(); i++)
	{
		// get 128 dimensions in a vector
		vector<float> k1;
		for(int x=0; x<128; x++)
		{
			k1.push_back((float)this->descriptors.at<float>(i,x));
		}

		// loop through keypoints2.size
		for (int j=0; j<m2.keypoints.size(); j++)
		{
			double temp=0;
			// calculate euclidian distance
			for(int x=0; x<128; x++)
			{
				temp += (pow((k1[x] - (float)m2.descriptors.at<float>(j,x)), 2.0));
			}
			vec1[i].push_back((float)sqrt(temp)); // store distance for each keypoints in image2
			unsortedvec1[i] = vec1[i];
		}
		sort(vec1[i].begin(),vec1[i].end()); // sort the vector distances to get shortest distance

		// find position of the shortest distance
		int pos = (int)(find(unsortedvec1[i].begin(), unsortedvec1[i].end(), vec1[i][0]) - unsortedvec1[i].begin());

		// assign that matchin feature to DMatch variable dm1
		dm1.queryIdx = i;
		dm1.trainIdx = pos;
		dm1.distance = vec1[i][0];
		matches1.push_back(dm1);
		this->matches.push_back(dm1);
		//cout << pos << endl;
    }

	// crete two dimensional vector for storing euclidian distance
	vector< vector<float> > vec2, unsortedvec2;
	for (int i=0; i<m2.keypoints.size(); i++)
	{
		vec2.push_back(vector<float>()); // Add an empty row
		unsortedvec2.push_back(vector<float>());
	}

	// create vector of DMatch for storing matxhes point
	vector< DMatch > matches2;
	DMatch dm2;
	// loop through keypoints2.size
	for (int i=0; i<m2.keypoints.size(); i++)
	{
		// get 128 dimensions in a vector
		vector<float> k1;
		for(int x=0; x<128; x++)
		{
			k1.push_back((float)m2.descriptors.at<float>(i,x));
		}

		// loop through keypoints1.size
		for (int j=0; j<this->keypoints.size(); j++)
		{
			double temp=0;
			// calculate euclidian distance
			for(int x=0; x<128; x++)
			{
				temp += (pow((k1[x] - (float)this->descriptors.at<float>(j,x)), 2.0));
			}
			vec2[i].push_back((float)sqrt(temp)); // store distance for each keypoints in image1
			unsortedvec2[i] = vec2[i];
		}
		sort(vec2[i].begin(),vec2[i].end()); // sort the vector distances to get shortest distance

		// find position of the shortest distance
		int pos = (int)(find(unsortedvec2[i].begin(), unsortedvec2[i].end(), vec2[i][0]) - unsortedvec2[i].begin());

		// assign that matchin feature to DMatch variable dm1
		dm2.queryIdx = i;
		dm2.trainIdx = pos;
		dm2.distance = vec2[i][0];
		matches2.push_back(dm2);
		m2.matches.push_back(dm2);

		//cout << pos << endl;
    }

	//-- Quick calculation of max and min distances between keypoints1
	double max_dist = 0;
	double min_dist = 500.0;
	for( int i = 0; i < matches1.size(); i++ )
	{
		double dist = matches1[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	//printf("-- Max dist : %f \n", max_dist );
	//printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches1 (i.e. whose distance is less than 2*min_dist )
	//-- PS.- radiusMatch can also be used here.
	int MatchingFactor = 5; // Gives a good result [3-5] range
	double distanceFactor = 0.8; // Gives a good resut [0.5-0.8]range
	std::vector< DMatch > good_matches1;
	for( int i = 0; i < matches1.size(); i++ )
	{
		if( matches1[i].distance <= max(MatchingFactor*min_dist, distanceFactor) )
		{
			good_matches1.push_back( matches1[i]);
		}
		//if( matches1[i].distance <= 2*min_dist )
		//{
			//good_matches1.push_back( matches1[i]);
		//}
	}

	//-- Quick calculation of max and min distances between keypoints2 but not used
	for( int i = 0; i < matches2.size(); i++ )
	{
		double dist = matches2[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	//printf("-- Max dist : %f \n", max_dist );
	//printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches by comparing that (ft1 gives ft2) and (ft2 gives ft1)
	std::vector< DMatch > good_matches;
	for(unsigned int i=0; i<good_matches1.size(); i++)
	{
		// check ft1=ft2    and ft2=ft1
		if(good_matches1[i].queryIdx == matches2[good_matches1[i].trainIdx].trainIdx)
			good_matches.push_back(good_matches1[i]);
	}
	//-----------------------------------------------------------------------------------------------------------------//
	return good_matches;
}

void Assignment2::displayFeatures(Assignment2 &m2)
{
	Size sz1 = this->image.size(); // find size of image1
    Size sz2 = m2.image.size(); // find size of image2
	Mat output((sz1.height>sz2.height)?sz1.height:sz2.height, sz1.width+sz2.width, CV_8UC3); // create output image to display two images in one window

	drawKeypoints(this->image, this->keypoints, this->sift_output, Scalar(0,0,255));  // draw features
	drawKeypoints(m2.image, m2.keypoints, m2.sift_output, Scalar(0,0,255));  // draw features

	// set leftROI and rightROI on that output image for display
	Mat left(output, Rect(0, 0, sz1.width, sz1.height));
	this->sift_output.copyTo(left);
    Mat right(output, Rect(sz1.width, 0, sz2.width, sz2.height));
	m2.sift_output.copyTo(right);

	// display output image with feature displayed
	imwrite("SIFT-Feature-Matching.jpg", output);
	//imshow("SIFT Feature Matching", output);
}

void Assignment2::displayGoodMatches(Assignment2 &m2, const vector< DMatch> &good_matches)
{
	// draw matches defined in good matches
	Mat img_matches;
	drawMatches( this->image, this->keypoints, m2.image, m2.keypoints, good_matches, img_matches, Scalar(0,0,255),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//-- Show detected matches
	cout <<" Good Matches Size : "<<good_matches.size() << endl;
	imwrite( "Good-Matches.jpg", img_matches );
	//imshow( "Good Matches", img_matches );
}

Mat Assignment2::computeRANSAC_opencv(Assignment2 &m2) // used openCV function to counter check with my calculated H
{
	vector< DMatch > good_matches=this->FindMatchesEuclidian(m2);

	vector<Point2f> query;
	vector<Point2f> train;

	for(unsigned int i=0; i<(good_matches.size()); i++)
	{
		query.push_back(this->keypoints[good_matches[i].queryIdx].pt);
		train.push_back(m2.keypoints[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(query, train, CV_RANSAC, PIXEL_DIST);

	cout<<"Homography"<<std::endl;
	cout<<H<<std::endl;

	return H;
}

// Use RANSAC to compute H without openCV function
Mat Assignment2::computeRANSAC(Assignment2 &m2)
{
	vector< DMatch > matches = this->FindMatchesEuclidian(m2); // find matches based on euclidian distances

	// Create vectors for storing Points
	vector<Point2f> query(matches.size());
	vector<Point2f> train(matches.size());

	//cout << matches.size() << " " << keypoints.size() << endl;

	// Assign points to that declared array
	for(unsigned int i=0;i<(matches.size());i++)
	{
		query[i]=this->keypoints[matches[i].queryIdx].pt;
		train[i]=m2.keypoints[matches[i].trainIdx].pt;
	}

	// define parameters for RANSAC algorithm
	int iterations=0;
	double inlierThresh=0.90;
	double inlierPerc=0;
	Mat H;
	Mat Hbest;
	double inlierbest=0;

	// do while loop for 3000 iterations
	do
	{
		int N=matches.size();
		int M=4;
		vector<Point2f> queryH(M);  // take 4 points
		vector<Point2f> trainH(M);
		vector<int> randVecId=randomGenerate(N,M);

		for(int i=0;i<M;i++)
		{
			queryH[i]=query[randVecId[i]]; // take 4 points randomly
			trainH[i]=train[randVecId[i]];
		}

		H = computeH(queryH, trainH);  // Compute H based on the that 4 points

		double inliners = computeInliers(train, query, H, 3);  // calculate inliners for that H
		cout<<"No of Inliers : "<<inliners<<std::endl ;

		if(inliners>inlierbest)
		{
			inlierbest=inliners;
			Hbest=H.clone();
		}


		inlierPerc=inliners/(double)N;

		iterations++;

	} while(iterations<3000);//&&inlierPerc<inlierThresh);
    //cout<<"No of Inliers : "<<inliners<<std::endl ;

	normalizeH(Hbest);  // normalize Hbest

	return Hbest;
}

/*Reference: http://stackoverflow.com/questions/1608181/unique-random-numbers-in-an-integer-array-in-the-c-programming-language */
// Used to calculate random generation of points
vector<int> Assignment2::randomGenerate(int N, int M)//Flyodd Algo
{
	vector<unsigned char> is_used(N, 0); /* flags */
	int in, im;
	vector<int> vektor(M);
	im = 0;

	for (in = N - M; in < N && im < M; ++in) {
	  int r = rand() % (in + 1); /* generate a random number 'r' */

	  if (is_used[r])
	    /* we already have 'r' */
	    r = in; /* use 'in' instead of the generated number */

	  assert(!is_used[r]);
	  vektor[im++] = r + 1; /* +1 since your range begins from 1 */
	  is_used[r] = 1;
	}

	assert(im == M);
	return vektor;
}

Mat Assignment2::computeH(const vector<Point2f> &query, const vector<Point2f> &train)
{
	Mat I1(3,4,CV_64FC1,0.0);
	Mat I2(3,4,CV_64FC1,0.0);
	Mat H(3,3,CV_64FC1,0.0);

	for(unsigned int i=0; i<(query.size()); i++)
	{
		// Assign co-ordinates to the matrix
		I1.at<double>(0,i)=query[i].x;
		I1.at<double>(1,i)=query[i].y;
		I1.at<double>(2,i)=1;

		I2.at<double>(0,i)=train[i].x;
		I2.at<double>(1,i)=train[i].y;
		I2.at<double>(2,i)=1;
	}

	// solve linear equations
	solve(I1.t(), I2.t(), H, DECOMP_SVD);

	H = H.t();

	return H;
}

int Assignment2::computeInliers(const vector<Point2f> &train, const vector<Point2f> &query, Mat &H, double thresh)
{
	Mat Hmat=H.clone();

	int sumInliers=0;
	for(unsigned int i=0; i<(train.size()); i++)
	{
		// calculate inliners based on the H and euclidian distances
		Mat pt(3,1,CV_64FC1,0.0);
		pt.at<double>(0,0)=query.at(i).x;
		pt.at<double>(1,0)=query.at(i).y;
		pt.at<double>(2,0)=1;

		Mat proj= Hmat*pt;
		proj.at<double>(0,0)=proj.at<double>(0,0)/proj.at<double>(2,0);
		proj.at<double>(1,0)=proj.at<double>(1,0)/proj.at<double>(2,0);
		proj.at<double>(2,0)=proj.at<double>(2,0)/proj.at<double>(2,0);

		double res;
		double temp1=(proj.at<double>(0,0)-train.at(i).x);
		double temp2=(proj.at<double>(1,0)-train.at(i).y);
		res = (temp1*temp1) + (temp2*temp2);
		res = sqrt(res);

		// if distance is less than threshold then it is a good fit for that point
		if(res <= thresh)
		{
			sumInliers++;
		}
	}
    cout<<"Total number of inliers ... "<<sumInliers<<endl;
	return sumInliers;
}

void Assignment2::normalizeH(Mat Hbest)
{
	// Normalizes Hbest
	Hbest.at<double>(0,0)/=Hbest.at<double>(2,2);
	Hbest.at<double>(0,1)/=Hbest.at<double>(2,2);
	Hbest.at<double>(0,2)/=Hbest.at<double>(2,2);
	Hbest.at<double>(1,0)/=Hbest.at<double>(2,2);
	Hbest.at<double>(1,1)/=Hbest.at<double>(2,2);
	Hbest.at<double>(1,2)/=Hbest.at<double>(2,2);
	Hbest.at<double>(2,0)/=Hbest.at<double>(2,2);
	Hbest.at<double>(2,1)/=Hbest.at<double>(2,2);
	Hbest.at<double>(2,2)/=Hbest.at<double>(2,2);
}

Mat Assignment2::warpImage(Mat H)
{
	// Apply perspective transformation to see effect of H
	Mat hinv = H.inv(DECOMP_LU);
	Mat translated_image;

	warpPerspective(this->image, translated_image, hinv, this->image.size(), CV_INTER_LINEAR, BORDER_CONSTANT);

	return translated_image;
}

Mat Assignment2::displayOverlaying(Assignment2 &m2, Mat H, Mat image)
{

	//-- Quick calculation of max and min distances between keypoints1
	double max_dist = 0;
	double min_dist = 500.0;
	for( int i = 0; i < m2.matches.size(); i++ )
	{
		double dist = m2.matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	//printf("-- Max dist : %f \n", max_dist );
	//printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches1 (i.e. whose distance is less than 2*min_dist )
	//-- PS.- radiusMatch can also be used here.
	int MatchingFactor = 5; // Gives a good result [3-5] range
	double distanceFactor = 0.8; // Gives a good resut [0.5-0.8]range
	std::vector< DMatch > good_matches2;
	for( int i = 0; i < m2.matches.size(); i++ )
	{
		if( m2.matches[i].distance <= max(MatchingFactor*min_dist, distanceFactor) )
		{
			good_matches2.push_back( m2.matches[i]);
		}
		//if( m2.matches[i].distance <= 4*min_dist )
		//{
		//	good_matches2.push_back(m2.matches[i]);
		//}
	}

	// Define two matrix for calculating overlaying
	Mat I1(3,good_matches2.size(),CV_64FC1,0.0);
	Mat I2(3,good_matches2.size(),CV_64FC1,0.0);

	for(unsigned int i=0; i<good_matches2.size(); i++)
	{
		// Assign co-ordinates to the matrix
		I2.at<double>(0,i) = m2.keypoints[good_matches2[i].trainIdx].pt.x;
		I2.at<double>(1,i) = m2.keypoints[good_matches2[i].trainIdx].pt.y;
		I2.at<double>(2,i) = 1;
	}

	// Solve for the co-ordinates in image 1
	solve(H, I2, I1);

	// draw red circle around keypoints in image1
	for(unsigned int i=0; i<this->keypoints.size(); i++)
	{
		Point2f center = Point2f(this->keypoints[this->matches[i].queryIdx].pt.x, this->keypoints[this->matches[i].queryIdx].pt.y);
		circle(image, center, 1, Scalar(0,0,255), 2);
	}

	// draw green circle around overlaying keypoints
	for(int i=0; i<I1.cols; i++)
	{
		Point2f center = Point2f(I1.at<double>(0,i), I1.at<double>(1,i));
		circle(image, center, 1, Scalar(0,255,0), 2);
	}
	//-- Show detected matches
	//cout << good_matches.size() << endl;
	//imwrite( "Good-Matches.jpg", img_matches );

	return image;
}
