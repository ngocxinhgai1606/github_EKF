//#include <QCoreApplication>
#include <QApplication>
#include <iostream>
#include "KalmanLRF.h"
#include <QDebug>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{

    float frequencyLaser = 30; //hz
    float dt = 1.0/frequencyLaser ; // khoảng thời gian giữa hai dữ liệu liên tiếp
    int measurementSize = 4643;    // data size
    int groundDataSize = measurementSize;
    int frequencyFrame = 30; //(fps)
    int timeLate = 300; //ms
    int predictNumberValues = 9; //(frequencyFrame*timeLate/1000);

    string measurementPath = "/home/ngocxinhgai/Documents/QT projects/EKF_CV/Data/TB1_1609_075323.txt";
    string outputPath = "/home/ngocxinhgai/Documents/QT projects/EKF_CV/Data/TB1_1609_075323_out.csv";

//    string measurementPath = "/media/ngocxinhgai/01D5F8A42743AFD0/Documents/PK57/Task5_laser_30fps/python/results/2023_03_14/follow_f/230314_input_cplus03.txt";
//    string outputPath = "/media/ngocxinhgai/01D5F8A42743AFD0/Documents/PK57/Task5_laser_30fps/python/results/2023_03_14/follow_f/230314_output_cplus03.csv";


    Mat grountTruth = Mat::zeros(groundDataSize, 4, CV_32F);
//    readDataFromTextFile(groundtruth_path.c_str(), grount_truth.data, DATATYPE_FLOAT);
    Mat measurements = Mat::zeros(measurementSize, 3, CV_32F);
    readDataFromTextFile(measurementPath.c_str(), measurements.data, DATATYPE_FLOAT);
    Mat estimatedRange = Mat::zeros(measurementSize, 3, CV_32F);
    Mat estimatedUpdate = Mat::zeros(measurementSize, 6, CV_32F);

    KalmanLRF kalman_lrf;

    Mat measurement1 = Mat::zeros(1, 3, CV_32F);
    float deltat = 0;
    double rangeErrorEs = 5; //m
    double aziErrorEs = 0.3 / 1000; //rad
    double eleErrorEs = 0.3 / 1000; //rad
    bool lostTrack = false;   //giá trị từ laser trả về
    float objectVmax = 80; // the maximum speed of the considered object (m/s)
    kalman_lrf.setParams(1 / dt, rangeErrorEs, aziErrorEs, eleErrorEs);
    /*
    Start this feature when continuous laser range finder was on
    When restart continuous laser range finder, rerun the whole process
    */

    double x;
    double y;
    double z;
    for (uint i = 0; i < measurements.rows; i++)
    {

        //Run one loop when a measurement of laser was returnedd

        Mat measurement = measurements.row(i);	//Measurement: range (return from laser measurement), elevation, azimuth
        lostTrack = false;
//        cout <<"i   " << i  <<measurement << endl;
        if (measurement.at<float>(0) ==0)
        {   lostTrack = true;
        }
        // convert distance, tilt, pan sang tọa độ điểm x,y,z trong không gian 3 chiều, tâm tại đài
        x = measurement.at<float>(0) * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
        y = measurement.at<float>(0) * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
        z = measurement.at<float>(0) * sin(measurement.at<float>(1));

        grountTruth.row(i).at<float>(0) =x ;
        grountTruth.row(i).at<float>(1) =y ;
        grountTruth.row(i).at<float>(2) =z ;
        grountTruth.row(i).at<float>(3) = sqrt( x*x +y*y +z*z);

        if (measurement1.at<float>(0) == 0 && measurement.at<float>(0) != 0)
        {
            printf("init\n");
            measurement.copyTo(measurement1);	//Capture the 1st valid measurement - all attributes are not zero
            deltat = i* dt;
            kalman_lrf.init(measurement1, deltat);
            estimatedRange.row(i).at<float>(0) =x ;
            estimatedRange.row(i).at<float>(1) =y ;
            estimatedRange.row(i).at<float>(2) =z ;
        }

        else if (measurement1.at<float>(0) != 0)
        {
            if (lostTrack)
            {
                printf("predict\n");
//                Mat predictedState = kalman_lrf.predict();
                Mat predictedState = kalman_lrf.predict(predictNumberValues);
                estimatedRange.row(i).at<float>(0) = predictedState.at<float>(0);
                estimatedRange.row(i).at<float>(1) = predictedState.at<float>(1);
                estimatedRange.row(i).at<float>(2) = predictedState.at<float>(2);
            }
            else
            {
                printf("update\n");
                deltat = i*dt;
                estimatedUpdate = kalman_lrf.update(measurement, deltat, objectVmax,predictNumberValues);
                estimatedRange.row(i).at<float>(0) = estimatedUpdate.at<float>(0);
                estimatedRange.row(i).at<float>(1) = estimatedUpdate.at<float>(1);
                estimatedRange.row(i).at<float>(2) = estimatedUpdate.at<float>(2);
            }
        }
        else
        {
            estimatedRange.row(i).at<float>(0) = grountTruth.row(0).at<float>(0) ;
            estimatedRange.row(i).at<float>(1) = grountTruth.row(0).at<float>(1) ;
            estimatedRange.row(i).at<float>(2) = grountTruth.row(0).at<float>(2) ;
        }
    }

    std::cout << "estimated_range.rows: " << estimatedRange.rows << endl;

    std::ofstream output_file;
    output_file.open (outputPath);
    output_file << "dis_gr, dis_pr, tilt_gr, tilt_pr, pan_gr, pan_pr" << endl;

    for (uint i = 0; i < estimatedRange.rows; i++) {
        x = estimatedRange.row(i).at<float>(0);
        y = estimatedRange.row(i).at<float>(1);
        z = estimatedRange.row(i).at<float>(2);
        double estimatedDistance = sqrt( x*x +y*y +z*z);
        double tiltPre = asin(z/estimatedDistance);
        //pan = math.acos(y/(distance * math.cos(tilt)))
        float panPre = atan2(x,y);
//        if (panPre < 0) {
//            panPre = panPre + 2 * M_PI;
//        }
        output_file << measurements.row(i).at<float>(0) << "," <<  estimatedDistance << "," <<  measurements.row(i).at<float>(1) << "," <<tiltPre << "," << measurements.row(i).at<float>(2)<<"," << panPre<< endl;
            }
    output_file.close();
}

    // Run program: Ctrl + F5 or Debug > Start Without Debugging menu
    // Debug program: F5 or Debug > Start Debugging menu

    // Tips for Getting Started:
    //   1. Use the Solution Explorer window to add/manage files
    //   2. Use the Team Explorer window to connect to source control
    //   3. Use the Output window to see build output and other messages
    //   4. Use the Error List window to view errors
    //   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
    //   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
