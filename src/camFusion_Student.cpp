
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/*
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size.
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{ 
  	
    std::vector<cv::DMatch> matches_selected;
    std::vector<double> matches_distance;

  	// check the keypoints are within the bounding boxes
    for (int i = 0; i < kptMatches.size(); ++i)
    {
      	auto distance=cv::norm(kptsCurr[kptMatches[i].trainIdx].pt - kptsPrev[kptMatches[i].queryIdx].pt);
        if (boundingBox.roi.contains(kptsCurr[kptMatches[i].trainIdx].pt))
        {	
            matches_selected.push_back(kptMatches[i]);
            matches_distance.push_back(distance);
        }
      	else{
        	continue;
        }
    }
  	
	// calculate the mean of the match distances to fileter the points
  	// https://stackoverflow.com/questions/28574346/find-average-of-input-to-vector-c
    double meanDistance = std::accumulate(matches_distance.begin(), matches_distance.end(), 0.0) / matches_distance.size();
  	
    for (int i = 0; i < matches_distance.size(); ++i)
    {
        if (matches_distance[i] < meanDistance) // if near to the mean it's okey else continue;
        {
            boundingBox.kptMatches.push_back(matches_selected[i]);
          	boundingBox.keypoints.push_back(kptsCurr[matches_selected[i].trainIdx]);
        }
      	else{
        	continue;
        }
    } 
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    } // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // use medianDistRatio is more robust than meanDistRatio
    std::sort(distRatios.begin(), distRatios.end());
    long medianIndex = floor(distRatios.size() / 2.0);
    double medianDistRatio = (distRatios.size() % 2 == 0) ? (distRatios[medianIndex-1] + distRatios[medianIndex]) / 2.0 : distRatios[medianIndex];

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
}


double calc_median(std::vector<LidarPoint> input_vec){
	double median_result = 0;

  	vector<double> x_vec;
  	for (int i =0;i<input_vec.size();++i)
    {
        x_vec.push_back(input_vec[i].x);
    }
	sort(x_vec.begin(), x_vec.end());
  	
    double median_idx = floor(x_vec.size() / 2.0);
    if (x_vec.size() % 2 == 0) 
        median_result = (x_vec[median_idx - 1] + x_vec[median_idx]) / 2.0;
    else
        median_result = x_vec[median_idx];
	return median_result;
}
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1 / frameRate;
  	double median_dist_prev = calc_median(lidarPointsPrev);
	double median_dist_curr = calc_median(lidarPointsCurr);
    TTC = median_dist_curr * dT / (median_dist_prev - median_dist_curr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
	int prev_size = prevFrame.boundingBoxes.size();
  	int curr_size = currFrame.boundingBoxes.size();
    int match_counts[prev_size][curr_size] = {0};

    for (int i = 0; i < matches.size(); ++i)
    {
      	std::vector<int> prev_bounding_index, curr_bounding_index;
        cv::KeyPoint prev_keypoint = prevFrame.keypoints[matches[i].queryIdx];
        for (int j = 0; j < prevFrame.boundingBoxes.size(); ++j)
        {
            if (prevFrame.boundingBoxes[j].roi.contains(prev_keypoint.pt))
            {
                prev_bounding_index.push_back(prevFrame.boundingBoxes[j].boxID);
            }
        }
		
      	cv::KeyPoint curr_keypoint = currFrame.keypoints[matches[i].trainIdx];
        for (int j = 0; j < currFrame.boundingBoxes.size(); ++j)
        {
            if (currFrame.boundingBoxes[j].roi.contains(curr_keypoint.pt))
            {
                curr_bounding_index.push_back(currFrame.boundingBoxes[j].boxID);
            }
        }

        for (int j = 0; j < prev_bounding_index.size(); ++j)
        {
            for (int k = 0; k < curr_bounding_index.size(); ++k)
            {
                match_counts[prev_bounding_index[j]][curr_bounding_index[k]]++;
            }
        }
    }

    for (int j = 0; j < prevFrame.boundingBoxes.size(); ++j)
    {
        int max_match_counts = 0;
        int max_index = 0;
        for (int k = 0; k < currFrame.boundingBoxes.size(); ++k)
        {
            if (match_counts[j][k] > max_match_counts)
            {
                max_match_counts = match_counts[j][k];
                max_index = k;
            }
        }
        bbBestMatches.insert({j, max_index});
    }
}