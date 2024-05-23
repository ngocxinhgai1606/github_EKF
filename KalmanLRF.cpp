#include "KalmanLRF.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <QDebug>

using namespace Eigen;
using namespace std;

void readDataFromTextFile(const char *path, void *data, const char type) {
    string line;
    ifstream dataFile(path);
    string token;
    uint32_t index = 0;
    if (type == DATATYPE_DECIMAL) {
        if (dataFile.is_open()) {
            while (getline(dataFile, line)) {
                stringstream sline(line);
                while (getline(sline, token, '\t')) {
                    ((int16_t *)data)[index] = stoi(token);
                    index++;
                }
            }
            dataFile.close();
        }
    } else if (type == DATATYPE_FLOAT) {
        if (dataFile.is_open()) {
            while (getline(dataFile, line)) {
                stringstream sline(line);
                while (getline(sline, token, '\t')) {
                    if (token.compare("\r") != 0)
                     {
                        ((float *)data)[index] = stof(token);

                        index++;
                    }

                }
            }
            dataFile.close();
        }
    } else {
        if (dataFile.is_open()) {
            while (getline(dataFile, line)) {
                stringstream sline(line);
                while (getline(sline, token, '\t')) {
                    ((char *)data)[index] = stoi(token);
                    index++;
                }
            }
            dataFile.close();
        }
    }
    return;
}

KalmanLRF::KalmanLRF() {
    KF.init(stateDim, measureDim, 0, CV_32F);
}

KalmanLRF::~KalmanLRF() {
}

void KalmanLRF::setParams(double samplingFreq, double rangeErrorEs, double aziErrorEs, double eleErrorEs, Mat accErrorEs) {
    this->rangeErrorEs = rangeErrorEs;
    this->aziErrorEs = aziErrorEs;
    this->eleErrorEs = eleErrorEs;
    this->dt = 1.0 / samplingFreq;
    this->samplingFreq = samplingFreq;
    // acc_error_es.copyTo(this->acc_error_es);
}

const Mat &KalmanLRF::predict() {
    return KF.predict();
}

Mat KalmanLRF::predict(int max) {
    //    qDebug() << "predict at " << count;
    cv::Mat predictValues = KF.predict();
    for(int count = 2; count <= max; count ++) {
        predictValues = KF.transitionMatrix * predictValues;
//        float x = predictValues.at<float>(0);
//        float y = predictValues.at<float>(1);
//        float z = predictValues.at<float>(2);
//        float distance =  sqrt(pow(x,2) + pow(y,2) + pow(z,2));
//        float tilt = asin(z/distance);
//        //pan = math.acos(y/(distance * math.cos(tilt)))
//        float pan = atan2(x,y);
//        cout << "distance_predict at "<< count << " :  " << distance << endl;
//        cout << "tilt_predict at " << count << " :  " << tilt << endl;
//        cout << "pan_predict at " << count << " :  " << pan << endl;
    }
    return predictValues;

}




Mat KalmanLRF::hJacobian(Mat input){

    float x =  float(input.at<float>(0));
    float y =  float(input.at<float>(1));
    float z =  float(input.at<float>(2));
    float r = sqrt(pow(x,2) + pow(y,2) + pow(z,2));

    double Hx[3][6] = {{(x/r), (y/r),(z/r), (0), ( 0), ( 0)},
                    {((-x*z)/(sqrt(pow(x,2)+pow(y,2))*pow(r,2))), ( (-y*z)/(sqrt(pow(x,2)+pow(y,2))*pow(r,2))), ( sqrt(pow(x,2)+pow(y,2))/pow(r,2)), ( 0), ( 0), (0)},
                    {(y/(pow(x,2)+pow(y,2))), ( -x/(pow(x,2)+pow(y,2))), ( 0), ( 0), ( 0), ( 0)}};

    Mat Hx_mat(3, 6, CV_64F); // Tạo một ma trận 3x6 với kiểu dữ liệu double
    memcpy(Hx_mat.data, Hx, stateDim/2 * stateDim * sizeof(double));
    Hx_mat.convertTo(Hx_mat, CV_32F);
    return Hx_mat;
}

Mat KalmanLRF::hx(Mat estimateValue){
    Mat hx = Mat::zeros(3, 1, CV_32F);
    float x = estimateValue.at<float>(0);
    float y = estimateValue.at<float>(1);
    float z = estimateValue.at<float>(2);
    hx.row(0).at<float>(0) =  sqrt(pow(x,2) + pow(y,2) + pow(z,2));
    hx.row(1).at<float>(0) = asin(z/hx.row(0).at<float>(0));
    //pan = math.acos(y/(distance * math.cos(tilt)))
    float pan = atan2(x,y);
    if (pan < 0) {
        pan = pan + 2 * M_PI;
    }
    hx.row(2).at<float>(0) = pan;
    return hx;
}

void KalmanLRF::init(Mat measurement,  float currentTime) {

    Mat xState = Mat::zeros(stateDim, 1, CV_32F);
    Mat transitionMatrix = Mat::zeros(stateDim, stateDim, CV_32F);
    Mat transitionMatrixCA = Mat::eye(9, 9, CV_32F);
    Mat processNoiseCov = Mat::zeros(stateDim, stateDim, CV_32F);
    Mat errorCovPost = Mat::zeros(stateDim, stateDim, CV_32F);
    measurementNoiseCov.row(0).at<float>(0) = 25;
    measurementNoiseCov.row(1).at<float>(1) = pow(0.0003, 2);
    measurementNoiseCov.row(2).at<float>(2) = pow(0.0003, 2);

    xState.at<float>(0) = measurement.at<float>(0) * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
    xState.at<float>(1) = measurement.at<float>(0) * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
    xState.at<float>(2) = measurement.at<float>(0) * sin(measurement.at<float>(1));
    xState.at<float>(3) = 0;
    xState.at<float>(4) = 0;
    xState.at<float>(5) = 0;


    // Update the position components
    transitionMatrixCA.at<float>(0, 3) = dt;
    transitionMatrixCA.at<float>(1, 4) = dt;
    transitionMatrixCA.at<float>(2, 5) = dt;
    transitionMatrixCA.at<float>(3, 6) = dt;
    transitionMatrixCA.at<float>(4, 7) = dt;
    transitionMatrixCA.at<float>(5, 8) = dt;

    transitionMatrixCA.at<float>(0, 6) = 1/2*dt*dt;
    transitionMatrixCA.at<float>(1, 7) = 1/2*dt*dt;
    transitionMatrixCA.at<float>(2, 8) = 1/2*dt*dt;




    std::cout << "transitionMatrixCA  A/F: " << transitionMatrixCA << endl;
//    qDebug() << "transitionMatrixCA: " << transitionMatrixCA;

    Eigen::MatrixXd Q(9, 9);
        Q << 25 * dt * dt, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 25 * dt * dt, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 6.25 * dt * dt, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 100000 * dt * dt, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 100000 * dt * dt, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 25 * dt * dt, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 100 * dt * dt, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 25 * dt * dt, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 25 * dt * dt;

    // Output the matrix Q
    std::cout << "Q: " << Q << std::endl;




    Mat temp1 = Mat::zeros(stateDim / 2, stateDim, CV_32F);
    Mat temp2 = Mat::zeros(stateDim / 2, stateDim, CV_32F);
    Mat temp3 = Mat::zeros(stateDim / 2, stateDim, CV_32F);
    hconcat(Mat::eye(stateDim / 2, stateDim / 2, CV_32F), dt * Mat::eye(stateDim / 2, stateDim / 2, CV_32F), temp1);
    hconcat(Mat::zeros(stateDim / 2, stateDim / 2, CV_32F), Mat::eye(stateDim / 2, stateDim / 2, CV_32F), temp2);
    vconcat(temp1, temp2, transitionMatrix);
    std::cout << "transitionMatrix  A/F: " << transitionMatrix << endl;
//    qDebug() << "transitionMatrix  A/F: " << transitionMatrix;


    hconcat(Mat::diag(0.25 * pow(dt, 4) * accErrorEs.mul(accErrorEs)), Mat::zeros(stateDim / 2, stateDim / 2, CV_32F), temp1);
    hconcat(Mat::zeros(stateDim / 2, stateDim / 2, CV_32F), Mat::diag(dt * dt * accErrorEs.mul(accErrorEs)), temp2);
    vconcat(temp1, temp2, processNoiseCov);
    std::cout << "processNoiseCov Q: " << processNoiseCov << endl;
    std::cout << "........."<<endl;

    errorCovPost =  10000 * cv::Mat::eye(stateDim, stateDim, CV_32F);
    setIdentity(KF.measurementMatrix);              //H
    transitionMatrix.copyTo(KF.transitionMatrix);  //A/F
    processNoiseCov.copyTo(KF.processNoiseCov);     //Q
    errorCovPost.copyTo(KF.errorCovPost);           //Pk
    xState.copyTo(KF.statePost);                    // xk
    measurementNoiseCov.copyTo(KF.measurementNoiseCov); //R
    xState.copyTo(lastMeasurement);
    lastTime = currentTime;
    lastDistance = measurement.at<float>(0);
    lastPoint = converttoxyz(measurement);
    predictValues = KF.predict();
}
Mat KalmanLRF::converttoxyz(Mat measurement){
    float rangeMeasured = measurement.at<float>(0);
    float x = rangeMeasured * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
    float y = rangeMeasured * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
    float z = rangeMeasured * sin(measurement.at<float>(1));
    Mat xyz = (Mat_<float>(3, 1) << x, y, z);
    return xyz;
}
// Tính khoảng cách giữa hai diểm: điểm hợp lệ gần nhất và điểm đang xét hiện tại
// mục đích: so sánh với khoảng cách lớn nhất và đối tượng đạt được trong cùng khoảng thời gian để xác định điểm bất thường
float KalmanLRF::computedistancetwopoint(Mat measurement){
//    cout << "measurement" << measurement<< endl;
//    cout << "lastpoint" << lastpoint<< endl;

    Mat currencePoint =  converttoxyz (measurement);
    float xLast = lastPoint.row(0).at<float>(0);
    float yLast = lastPoint.row(1).at<float>(0);
    float zLast = lastPoint.row(2).at<float>(0);
    float xCurrent = currencePoint.row(0).at<float>(0);
    float yCurrent = currencePoint.row(1).at<float>(0);
    float zCurrent = currencePoint.row(2).at<float>(0);
    return sqrt(pow((xCurrent - xLast),2) +pow((yCurrent - yLast),2) + pow((zCurrent - zLast),2));

}
Mat KalmanLRF::update(Mat measurement, float currentime, float objectVmax, int predictNumberValues) {

    Mat output;
    float deltaTime = currentime - lastTime; // khoảng thời gian giữa lần có giá trị hợp lệ đến thời điểm hiện tại
    float maxdistance = deltaTime * objectVmax;  // quảng đường lớn nhất mà đối tượng đi được trong khoảng thời gian delta_time
    float deltaDistance = computedistancetwopoint(measurement); // quảng đường đi được thực tế mà đối tượng đã di chuyện trong khoảng thời gian delta_time
    Mat xState = KF.statePre;
    // check input
    if ( measurement.at<float>(0) != 0 && measurement.at<float>(0) != lastDistance && deltaDistance <= maxdistance )
        {
        printf("update_dk1");
        Mat hX = hx(xState);
        cout <<"hX.row(2).at<float>(0): " << hX.row(2).at<float>(0) << endl;
        double deltaPan = (measurement.at<float>(2) -  hX.row(2).at<float>(0))*180/M_PI;

        if (deltaPan < -180) {
            deltaPan = deltaPan + 360;
            hX.row(2).at<float>(0) = measurement.at<float>(2) - deltaPan*M_PI/180;
        }
        else if (deltaPan > 180) {
            deltaPan = deltaPan - 360;
            hX.row(2).at<float>(0) = measurement.at<float>(2) - deltaPan*M_PI/180;
        }
        else {
            hX.row(2).at<float>(0) = hX.row(2).at<float>(0);
        }
        Mat deltHx = measurement.t() - hX;
        Mat Hk = hJacobian(xState);
        Mat Sk = Hk * KF.errorCovPre * Hk.t() + KF.measurementNoiseCov;
        Mat SkInvert;
        invert(Sk, SkInvert, DECOMP_SVD);

//        cout << "measurement" << measurement<< endl;
//        cout << "lastpoint" << lastpoint<< endl;


        KF.gain = KF.errorCovPre * Hk.t() * SkInvert; //K
        KF.errorCovPost = KF.errorCovPre - KF.gain * Hk* KF.errorCovPre;    //P
        KF.statePost = KF.statePost + KF.gain * deltHx;
        KF.statePost.copyTo(output);
        lastTime = currentime;
        lastPoint = converttoxyz(measurement);
        lastDistance = measurement.at<float>(0);
      }
    else
    {
        printf("update_dk2");
        float distance = sqrt(pow(predictValues.at<float>(0),2) +pow(predictValues.at<float>(1),2) + pow(predictValues.at<float>(2),2));
        float x = distance * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
        float y = distance * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
        float z = distance * sin(measurement.at<float>(1));
        output = (Mat_<float>(stateDim, 1) << x, y, z, predictValues.at<float>(3), predictValues.at<float>(4), predictValues.at<float>(5));
        }

    Mat predictValues = predict(predictNumberValues);

    return predictValues ;
}




//        CV_PROP_RW Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
//        CV_PROP_RW Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
//        CV_PROP_RW Mat transitionMatrix;   //!< state transition matrix (A)
//        CV_PROP_RW Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
//        CV_PROP_RW Mat measurementMatrix;  //!< measurement matrix (H)
//        CV_PROP_RW Mat processNoiseCov;    //!< process noise covariance matrix (Q)
//        CV_PROP_RW Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
//        CV_PROP_RW Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
//        CV_PROP_RW Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
//        CV_PROP_RW Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
