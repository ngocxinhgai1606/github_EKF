#pragma once
#include <stdio.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

#define DATATYPE_DECIMAL 0
#define DATATYPE_FLOAT 1
#define DATATYPE_CHAR 2
#define ELEM_SWAP(a, b)         \
    {                           \
        register float t = (a); \
        (a) = (b);              \
        (b) = t;                \
    }

using namespace cv;
using namespace std;

/*
for testing only
@brief Read data from text file
@in path Pointer to file path
@in data Pointer to save data
@param type Type of data (float, char or decimal)
*/
void readDataFromTextFile(const char* path, void* data, const char type);

class KalmanLRF {
   private:
    KalmanFilter KF;
    float quickSelect(float arr[], int n);
    void madCal(float& madVal, float& medianVal, float arr[], int n);

    int stateDim = 6;
    int measureDim = 3;
    float madFactor = 1.482602218505602f;
    double lastValidResultTime = 0;
    float lastTime = 0;
    Mat lastPoint;
    Mat predictValues;
    float dt = 1/30;
    double rangeEst;
    double lastrangeMeasure;
    float lastDistance;
    double samplingFreq = 1;
    double rangeErrorEs = 5;                            // m
    double aziErrorEs = 0.3 / 1000;                     // mrad
    double eleErrorEs = 0.3 / 1000;                     // mrad
    Mat accErrorEs = (Mat_<float>(3, 1) << 10, 10, 5);  // acc_x, acc_y, acc_z
    Mat lastValidMeasurement = Mat::zeros(stateDim / 2, 1, CV_32F);
    int numValid = 0;
    Mat lastMeasurement = Mat::zeros(stateDim, 1, CV_32F);
//    Mat meanError = Mat::zeros(measureDim, 1, CV_32F);
//    Mat varError = Mat::zeros(measureDim, 1, CV_32F);
    Mat measurementNoiseCov = Mat::zeros(measureDim, measureDim, CV_32F);
    float dLast[2048];
    int dLastCurrentIdx = 0;

   public:
    KalmanLRF();
    virtual ~KalmanLRF();
    const Mat& predict();
    Mat hJacobian(Mat input);
    Mat predict(int predictNumberValues );
    Mat hx(Mat estimate_value);
    Mat update(Mat measurement, float currenttime, float object_vmax, int predictNumberValues);
    Mat converttoxyz(Mat measurement);
    float computedistancetwopoint (Mat measurement);
    void init(Mat measurement1,  float currentTime);
    void setParams(double samplingFreq, double rangeErrorEs = 0, double aziErrorEs = 0, double eleErrorEs = 0, Mat accErrorEs = cv::Mat());
};
