#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";
const char* dst_window = "dst image";
/// Function header
void cornerHarris_demo (int, void*);
void my_cornerHarris (const Mat& src, Mat& dst, int blockSize, int apertureSize, double k, int borderType);
/** @function main */
int main (int argc, char** argv) {
    /// Load source image and convert it to gray
    // src = imread ("house.jpg", 1);
	src = imread("D:\\OneDrive - mails.tsinghua.edu.cn\\Learn\\Digital Image Process\\Experiments\\Experiment-5\\pic5.png", 1);
    cvtColor (src, src_gray, COLOR_BGR2GRAY);

    /// Create a window and a trackbar
    namedWindow (source_window, WINDOW_AUTOSIZE);
    createTrackbar ("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
    imshow (source_window, src);

    cornerHarris_demo (0, 0);

    waitKey (0);
    return (0);
}

/** @function cornerHarris_demo */
void cornerHarris_demo (int, void*) {

    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros (src.size(), CV_32FC1);

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    my_cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    /// Normalizing
    normalize (dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    //convertScaleAbs (dst_norm, dst_norm_scaled);
	dst_norm_scaled.create(dst.rows, dst.cols, CV_8UC3);
	Mat_<Vec3b> m = dst_norm_scaled;


	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			uchar v = dst_norm.at<float>(j, i);
			m(j, i) = Vec3b(v, v, v);
		}
	}

    /// Drawing a circle around corners
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int)dst_norm.at<float>(j, i) > thresh) {
                circle(dst_norm_scaled, Point(i, j), 5, Scalar(255, 0, 0), 2, 8, 0);
            }
        }
    }
    /// Showing the result
    namedWindow(corners_window, WINDOW_NORMAL);
    imshow(corners_window, dst_norm_scaled);
}

void my_cornerHarris(const Mat& src, Mat& dst, int blockSize, int apertureSize, double k, int borderType) {
    // Convert the source image to grayscale
    if (src.channels() > 1) {
        cvtColor(src, src, COLOR_BGR2GRAY);
    }

    // Calculate the derivatives using Sobel operator
    Mat dx, dy;
    Sobel(src, dx, CV_32F, 1, 0, apertureSize, 1, 0, borderType);
    Sobel(src, dy, CV_32F, 0, 1, apertureSize, 1, 0, borderType);

    // Calculate the products of derivatives at each pixel
    Mat dx2 = dx.mul(dx);
    Mat dy2 = dy.mul(dy);
    Mat dxy = dx.mul(dy);

    // Apply Gaussian filter to the products of derivatives
    // GaussianBlur(dx2, dx2, Size(blockSize, blockSize), 0, 0, borderType);
    // GaussianBlur(dy2, dy2, Size(blockSize, blockSize), 0, 0, borderType);
    // GaussianBlur(dxy, dxy, Size(blockSize, blockSize), 0, 0, borderType);
    //Apply mean filter to the products of derivatives
    blur(dx2, dx2, Size(blockSize, blockSize), Point(-1, -1), borderType);
    blur(dy2, dy2, Size(blockSize, blockSize), Point(-1, -1), borderType);
    blur(dxy, dxy, Size(blockSize, blockSize), Point(-1, -1), borderType);
    
    // Calculate the Harris response
    Mat det = dx2.mul(dy2) - dxy.mul(dxy);
    Mat trace = dx2 + dy2;
    dst = det - k * trace.mul(trace);

    // Normalize the response
    normalize(dst, dst, 0, 255, NORM_MINMAX, CV_32F);

    // Convert the response to 8-bit image
    dst.convertTo(dst, CV_8U);
}
    