//
// Created by Administrator on 2017/8/31.
//
#include<jni.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/objdetect.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<android/log.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CppUtil.h"

using namespace cv;
using namespace std;


#define TAG    "qianliliang" // 这个是自定义的LOG的标识
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__) // 定义LOGD类型
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__) // 定义LOGI类型
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__) // 定义LOGW类型
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__) // 定义LOGE类型
#define LOGF(...)  __android_log_print(ANDROID_LOG_FATAL, TAG, __VA_ARGS__) // 定义LOGF类型
Mat *pMatInRGBA;

class LaneDetect
{
public:
    Mat currFrame; //stores the upcoming frame
    Mat temp;      //stores intermediate results
    Mat temp2;     //stores the final lane segments

    int diff, diffL, diffR;
    int laneWidth;
    int diffThreshTop;
    int diffThreshLow;
    int ROIrows;
    int vertical_left;
    int vertical_right;
    int vertical_top;
    int smallLaneArea;
    int longLane;
    int  vanishingPt;
    float maxLaneWidth;

    //to store various blob properties
    Mat binary_image; //used for blob removal
    int minSize;
    int ratio;
    float  contour_area;
    float blob_angle_deg;
    float bounding_width;
    float bounding_length;
    Size2f sz;
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    RotatedRect rotated_rect;


    LaneDetect(Mat startFrame)
    {
        currFrame = startFrame;                                    //if image has to be processed at original size

//        LOGW("zhongguoshige haorizi");
//        currFrame = Mat(480,640,CV_8UC1,0.0);                        //initialised the image size to 320x480
//        resize(startFrame, currFrame, currFrame.size());             // resize the input to required size

        temp      = Mat(currFrame.rows, currFrame.cols, CV_8UC1,0.0);//stores possible lane markings
        temp2     = Mat(currFrame.rows, currFrame.cols, CV_8UC1,0.0);//stores finally selected lane marks

        vanishingPt    = currFrame.rows/2;                           //for simplicity right now
        ROIrows        = currFrame.rows - vanishingPt;               //rows in region of interest
        minSize        = 0.00020 * (currFrame.cols*currFrame.rows);  //min size of any region to be selected as lane
        maxLaneWidth   = 0.025 * currFrame.cols;                     //approximate max lane width based on image size
        smallLaneArea  = 7 * minSize;
        longLane       = 0.3 * currFrame.rows;
        ratio          = 4;

        //these mark the possible ROI for vertical lane segments and to filter vehicle glare
        vertical_left  = 2*currFrame.cols/5;
        vertical_right = 3*currFrame.cols/5;
        vertical_top   = 2*currFrame.rows/3;
        //*pMatInRGBA = currFrame;
//        namedWindow("lane",WINDOW_AUTOSIZE);
//        namedWindow("midstep", WINDOW_AUTOSIZE);
//        namedWindow("currframe", WINDOW_AUTOSIZE);
//        namedWindow("laneBlobs",WINDOW_AUTOSIZE);

        getLane();
    }

    void updateSensitivity()
    {
        int total=0, average =0;
        for(int i= vanishingPt; i<currFrame.rows; i++)
            for(int j= 0 ; j<currFrame.cols; j++)
                total += currFrame.at<uchar>(i,j);
        average = total/(ROIrows*currFrame.cols);
        cout<<"average : "<<average<<endl;
    }

    void getLane()
    {
        //medianBlur(currFrame, currFrame,5 );
        // updateSensitivity();
        //ROI = bottom half
        for(int i=vanishingPt; i<currFrame.rows; i++)
            for(int j=0; j<currFrame.cols; j++)
            {
                temp.at<uchar>(i,j)    = 0;
                temp2.at<uchar>(i,j)   = 0;
            }

        //imshow("currframe", currFrame);
        blobRemoval();
    }

    void markLane()
    {
        for(int i=vanishingPt; i<currFrame.rows; i++)
        {
            //IF COLOUR IMAGE IS GIVEN then additional check can be done
            // lane markings RGB values will be nearly same to each other(i.e without any hue)

            //min lane width is taken to be 5
            laneWidth =5+ maxLaneWidth*(i-vanishingPt)/ROIrows;
            for(int j=laneWidth; j<currFrame.cols- laneWidth; j++)
            {

                diffL = currFrame.at<uchar>(i,j) - currFrame.at<uchar>(i,j-laneWidth);
                diffR = currFrame.at<uchar>(i,j) - currFrame.at<uchar>(i,j+laneWidth);
                diff  =  diffL + diffR - abs(diffL-diffR);

                //1 right bit shifts to make it 0.5 times
                diffThreshLow = currFrame.at<uchar>(i,j)>>1;
                //diffThreshTop = 1.2*currFrame.at<uchar>(i,j);

                //both left and right differences can be made to contribute
                //at least by certain threshold (which is >0 right now)
                //total minimum Diff should be atleast more than 5 to avoid noise
                if (diffL>0 && diffR >0 && diff>5)
                    if(diff>=diffThreshLow /*&& diff<= diffThreshTop*/ )
                        temp.at<uchar>(i,j)=255;
            }
        }

    }

    void blobRemoval()
    {
        markLane();
        LOGD("this is qianliliang");
//    vector<Vec4i> lines;
//    //霍夫直线检测
//    HoughLinesP(binary_image, lines, 1, CV_PI / 100, 15, 15, 10);
//    LOGW("lengthhahaha = %d", lines.size());
//    LOGW("mkmk = %i", cpputil_add() );
//
//    for (size_t i = 0; i < lines.size(); i++) {
//        line(*pMatInRGBA, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]),
//             Scalar(0, 255, 0), 5, 8);
//    }

        // find all contours in the binary image
        temp.copyTo(binary_image);
        findContours(binary_image, contours,
                     hierarchy, CV_RETR_CCOMP,
                     CV_CHAIN_APPROX_SIMPLE);
//        for (size_t i = 0; i < hierarchy.size(); i++) {
//            line(*pMatInRGBA, Point(hierarchy[i][0], hierarchy[i][1]), Point(hierarchy[i][2], hierarchy[i][3]),
//                 Scalar(255, 0, 0), 5, 8);
//        }
        // for removing invalid blobs
        if (!contours.empty())
        {

            for (size_t i=0; i<contours.size(); ++i)
            {
                //====conditions for removing contours====//

                contour_area = contourArea(contours[i]) ;
//                vector<Vec4i> lines;
//                HoughLinesP(contours, lines, 1, CV_PI / 180, 80, 30, 10);
                //blob size should not be less than lower threshold
                if(contour_area > minSize)
                {
                    rotated_rect    = minAreaRect(contours[i]);
                    sz              = rotated_rect.size;
                    bounding_width  = sz.width;
                    bounding_length = sz.height;


                    //openCV selects length and width based on their orientation
                    //so angle needs to be adjusted accordingly
                    blob_angle_deg = rotated_rect.angle;
                    LOGW("blob_angle_deg %d", blob_angle_deg);
                    if (bounding_width < bounding_length)
                        blob_angle_deg = 90 + blob_angle_deg;

                    //if such big line has been detected then it has to be a (curved or a normal)lane
                    if(bounding_length>longLane || bounding_width >longLane)
                    {
                        drawContours(currFrame, contours,i, Scalar(255), CV_FILLED, 8);
                        drawContours(*pMatInRGBA, contours,i, Scalar(0,0,255), CV_FILLED, 8);//蓝色
                    }

                        //angle of orientation of blob should not be near horizontal or vertical
                        //vertical blobs are allowed only near center-bottom region, where centre lane mark is present
                        //length:width >= ratio for valid line segments
                        //if area is very small then ratio limits are compensated
                    else if ((blob_angle_deg <-10 || blob_angle_deg >-10 ) &&
                             ((blob_angle_deg > -70 && blob_angle_deg < 70 ) ||
                              (rotated_rect.center.y > vertical_top &&
                               rotated_rect.center.x > vertical_left && rotated_rect.center.x < vertical_right)))
                    {

                        if ((bounding_length/bounding_width)>=ratio || (bounding_width/bounding_length)>=ratio
                            ||(contour_area< smallLaneArea &&  ((contour_area/(bounding_width*bounding_length)) > .75) &&
                               ((bounding_length/bounding_width)>=2 || (bounding_width/bounding_length)>=2)))
                        {
                            drawContours(currFrame, contours,i, Scalar(255), CV_FILLED, 8);
                            drawContours(*pMatInRGBA, contours,i, Scalar(0,255,0), CV_FILLED, 8);//绿色
                        }
                    }
                }
            }
        }
        //imshow("midstep", temp);
        //imshow("laneBlobs", temp2);
        //imshow("lane",currFrame);
        //*pMatInRGBA = temp2;

    }


    void nextFrame(Mat &nxt)
    {
        //currFrame = nxt;                        //if processing is to be done at original size

        resize(nxt ,currFrame, currFrame.size()); //resizing the input image for faster processing
        getLane();
    }

    Mat getResult()
    {
        return temp2;
    }

};//end of class LaneDetect


extern "C"
{
CascadeClassifier face_cascade;

//java层输入每一帧图片
JNIEXPORT void JNICALL Java_com_clb_school_opencv_1_1ndk_view_CameraActivity_processFrames
        (JNIEnv *env, jobject instance, jlong addrInRGBA, jlong addrOut) {
    pMatInRGBA = (Mat *) addrInRGBA;
    Mat *pMatOut = (Mat *) addrOut;
    Mat BGZimg, imageGray, imageBlur, imageEr, imageCanny, imageROI;

    (*pMatInRGBA).copyTo(BGZimg);
//不规则处理
    Point points[1][6];
    points[0][0] = Point(BGZimg.cols / 4, BGZimg.rows / 2);
    points[0][1] = Point(BGZimg.cols * 3 / 4, BGZimg.rows / 2);
    points[0][2] = Point(BGZimg.cols, BGZimg.rows);
    points[0][3] = Point(BGZimg.cols, 0);
    points[0][4] = Point(0, 0);
    points[0][5] = Point(0, BGZimg.rows);
    const Point *pt[1] = {points[0]};
    int npt[1] = {6};
    polylines(BGZimg, pt, npt, 1, 1, Scalar(0, 0, 0, 0), 1, 8, 0);
    fillPoly(BGZimg, pt, npt, 1, Scalar(0, 0, 0));

//灰度化图片
    cvtColor(BGZimg, imageGray, COLOR_BGR2GRAY);
    LaneDetect detect(imageGray);

//    detect.nextFrame(*pMatInRGBA);
//
////高斯模糊
//    GaussianBlur(imageGray, imageBlur, Size(3, 3), 0, 0);
////二值化图像
//    threshold(imageBlur, imageEr, 200, 255, CV_THRESH_BINARY);
////边缘检测
//    Canny(imageEr, imageCanny, 64, 192, 3);
//
//    vector<Vec4i> lines;

//    imageROI = imageCanny.adjustROI(imageCanny.size().height * 5 / 8, imageCanny.size().height * 0 / 8,
//                                    imageCanny.size().height * 3 / 8, imageCanny.size().height * 5 / 8);

//霍夫直线检测
//    HoughLinesP(imageROI, lines, 1, CV_PI / 180, 80, 30, 10);
//    LOGW("length = %d", lines.size());
//    LOGW("lengthhahaha = %i", cpputil_add() );
//
//    for (size_t i = 0; i < lines.size(); i++) {
//        line(*pMatInRGBA, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]),
//             Scalar(0, 255, 0), 5, 8);
//    }

    //*pMatInRGBA = imageROI;
    //=====================================================
    LOGW("length =nihao");
//
//    if (face_cascade.empty()) {
//        bool a = face_cascade.load("/mnt/sdcard/cars.xml");
//        LOGW("length = %d", a);
//    }
//
////灰度化图片
//    cvtColor(BGZimg, imageGray, COLOR_BGR2GRAY);
//    vector<Rect> rect;
//// Detect cars
//    face_cascade.detectMultiScale(imageGray, rect, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
//

//Draw border
//    for (int i = 0; i < rect.size();i++)
//    {
//        Point  center;
//        int radius;
//        center.x = cvRound((rect[i].x + rect[i].width * 0.5));
//        center.y = cvRound((rect[i].y + rect[i].height * 0.5));
//
//        radius = cvRound((rect[i].width + rect[i].height) * 0.25);
//        circle(*pMatInRGBA, center, radius, Scalar(0, 0, 255), 2);
//    }

    *pMatOut = imageBlur;
}
}

