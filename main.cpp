#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>



/** @function main */


//#include "AD_Util.h"
using namespace std;
using namespace cv;

void thresholding(Mat&src,Mat&dst){
    int iLowH = 0;
    int iHighH = 255;
    
    int iLowS = 0;   //smaller S lighter it can go
    int iHighS = 20;
    int iCompS = 30;

    int iLowV = 250;    //smaller V blacker it can go
    int iHighV = 255;
    int iCompV = 240;

    Mat imgHSV;
    vector<Mat> hsvSplit;
    Mat imgGray;
    cvtColor(src, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    cvtColor(src, imgGray, COLOR_BGR2GRAY);
    //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
    split(imgHSV, hsvSplit);
    equalizeHist(hsvSplit[2],hsvSplit[2]);
    merge(hsvSplit,imgHSV);
    Mat imgThresholded,imgThresholded2,imgThresholded3;
    
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
    inRange(imgGray, 80, 255, imgThresholded2);
    inRange(imgHSV, Scalar(iLowH, iLowS, iCompV), Scalar(iHighH, iHighS, iHighV), imgThresholded3);
    //imgThresholded=imgThresholded2;  //comment this like to switch from gray scale to hsv
     //Threshold the image
    Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
    
    //闭操作 (连接一些连通域)
    morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
    dst=imgThresholded;
}

void transformImg(Mat&src,Mat&dst, int value){
	Point2f inputQuad[4]; 
    // Output Quadilateral or World plane coordinates
    Point2f outputQuad[4];
         
    // Lambda Matrix
    Mat lambda( 2, 4, CV_32FC1 );
    //Input and Output Image;
    Mat input, output;
     int coValue=0;
     if(value<0){
        coValue=-value;
     }
    //Load the image
    input = src;
    // Set the lambda matrix the same type and size as input
    lambda = Mat::zeros( input.rows, input.cols, input.type() );
 
    // The 4 points that select quadilateral on the input , from top-left in clockwise order
    // These four pts are the sides of the rect box used as input 
    inputQuad[0] = Point2f( 0,0 );
    inputQuad[1] = Point2f( input.cols,0);
    inputQuad[2] = Point2f( input.cols-coValue,input.rows+1);
    inputQuad[3] = Point2f( +coValue,input.rows+1  );  
    // The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = Point2f( 0,0 );
    outputQuad[1] = Point2f( input.cols-1,0);
    outputQuad[2] = Point2f( input.cols-value,input.rows-1);
    outputQuad[3] = Point2f( value,input.rows-1  );
 
    // Get the Perspective Transform Matrix i.e. lambda 
    lambda = getPerspectiveTransform( inputQuad, outputQuad );
    // Apply the Perspective Transform just found to the src image
    warpPerspective(input,output,lambda,output.size() );
    dst=output;
}

void findDrawContours(Mat&src,Mat&dst){
    src.copyTo(dst);
    dst.setTo(0);
    vector<vector<Point> > contours;
    Mat cannyed;
    Canny(src,cannyed,240,253);
    findContours(cannyed, contours, RETR_LIST, CHAIN_APPROX_NONE);
    for(int i=0;i<contours.size();i++){
        if(contours[i].size()<150){
            continue;
        }
        drawContours(dst,contours,i,255,3);
        
    }
    
}

void drawFit(vector<vector<Point>> contours,Mat&dst){
    for(int i=0;i<contours.size();i++){
        if(contours[i].size()<80){
            continue;
        }
        Vec4f rst;
        fitLine(contours[i], rst, CV_DIST_L2, 0, 0.01,0.01);
        //y=(dy/dx)*x+c;
        //rst[3]=(rst[1]/rst[0])*rst[2]+c;
        int c=rst[3]-(rst[1]/rst[0])*rst[2];
        //x=(y-c)/(rst[1]/rst[0])
        int y1=dst.rows;
        int y2=0;

        if(contours[i][0].y<dst.rows/3){
            y1=dst.rows/3;
        }else if(contours[i][0].y<dst.rows/3*2){
            y1=dst.rows/3*2;
            y2=dst.rows/3;
        }else{
            y2=dst.rows/3*2;
        }

        int x1=(y1-c)/(rst[1]/rst[0]);
        int x2=(y2-c)/(rst[1]/rst[0]);
        line(dst,Point(x1,y1),Point(x2,y2),Scalar(0,0,255),3);
    }
}


int main(int argc, char** argv){
    if( argc != 2)
    {
        cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }
    
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file
    
    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    resize(image, image, Size(640,480));
    image = image(Rect(0, image.rows / 2, image.cols, image.rows / 2));
    
    Mat contoursImg;
    Mat contoursImg2;
    //image.copyTo(contoursImg2);
    

    
    Mat bluredImg;
    //GaussianBlur( image, bluredImg, Size( 5, 5 ), 0, 0 );
    bluredImg=image;

    cvtColor(bluredImg,contoursImg,CV_BGR2GRAY);
    findDrawContours(contoursImg,contoursImg2);

    transformImg(bluredImg,bluredImg,286);
    bluredImg = bluredImg(Rect(0, 0, image.cols, image.rows -10));
    transformImg(contoursImg2,contoursImg2,286);
    
    line(contoursImg2,Point(0,image.rows/3),Point(image.cols,image.rows/3),0,5);
    line(contoursImg2,Point(0,image.rows/3*2),Point(image.cols,image.rows/3*2),0,5); 
    //cut the imag to 3 parts

    Mat binaryImg;
    thresholding(bluredImg,binaryImg);

    //transformImg(binaryImg,binaryImg);
    vector<vector<Point>> contours, contours2;
    findContours(binaryImg, contours, RETR_LIST, CHAIN_APPROX_NONE);
    findContours(contoursImg2, contours2, RETR_LIST, CHAIN_APPROX_NONE);

    cout<<contours.size()<<", "<<contours2.size()<<endl;
    drawFit(contours2,bluredImg);

    //put the marked img back to original
    Mat backToOriginal;
    transformImg(bluredImg,backToOriginal,-250);

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", bluredImg );
    
    namedWindow( "Display window binary", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window binary", binaryImg );
    // Show our image inside it.
    
    namedWindow( "Display window contour", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window contour", contoursImg2 );

    namedWindow( "Display window back to original view", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window back to original view", backToOriginal );


    waitKey(0);                                          // Wait for a keystroke in the window
    
    cout<<"RoboGrinders Wins"<<endl;
    return 0;
}
