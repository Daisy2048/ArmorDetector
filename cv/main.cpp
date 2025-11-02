#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
    VideoCapture capture("1351071190-1-160.mp4");
    if (!capture.isOpened())
    {
        cout << "Error! Please check your every step.";
        return 1;
    }

    double rate = capture.get(CAP_PROP_FPS);
    bool stop(false);
    Mat frame;

    namedWindow("Video Feed", WINDOW_NORMAL);
    resizeWindow("Video Feed", 1500, 1300);

    namedWindow("Contours", WINDOW_NORMAL);
    resizeWindow("Contours", 1500, 1300);

    namedWindow("Extracted Frame");
    int delay = 1000 / rate;

    while (!stop)
    {
        if (!capture.read(frame))
            break;
        imshow("Extracted Frame", frame);

        Mat image = frame.clone();

        if (image.empty())
        {
            cout << "404 not found";
            return -1;
        }

        Mat grayImage;
        cvtColor(image, grayImage, COLOR_BGR2GRAY);

        Mat binaryImage;
        threshold(grayImage, binaryImage, 127, 255, THRESH_BINARY);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        cout << "the num of contours:" << contours.size() << endl;
        Mat result_image = frame.clone();

        vector<RotatedRect> lightBars;
        
        for (size_t i = 0; i < contours.size(); i++)
        {
            double area = contourArea(contours[i]);
            
            // 面积筛选
            if (area < 150 || area > 3000) 
                continue;
            
            RotatedRect rect = minAreaRect(contours[i]);
            Size2f size = rect.size;
            
            float width = min(size.width, size.height);
            float height = max(size.width, size.height);
            
            float aspectRatio = height / width;
            
            // 角度筛选 - 新增部分
            float angle = rect.angle;
            
            // 角度调整：确保角度表示长边的方向
            if (size.width < size.height) {
                angle = 90 + angle;
            }
            
            // 筛选接近竖直的灯条：角度在60-120度之间
            if (fabs(angle) < 60 || fabs(angle) > 120) {
                continue;
            }
            
            // 宽高比和面积筛选
            if (aspectRatio > 3.0 && aspectRatio < 8.0 && 
                area > 150 && area < 3000)
            {
                lightBars.push_back(rect);
                
                Point2f vertices[4];
                rect.points(vertices);
                for (int j = 0; j < 4; j++)
                {
                    line(result_image, vertices[j], vertices[(j+1)%4], Scalar(0, 255, 0), 3);
                }
                
                // 可选：显示角度信息
                string angle_text = "A:" + to_string((int)angle);
                putText(result_image, angle_text, Point(rect.center.x-20, rect.center.y-15), 
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
            }
        }

        // 收集所有灯条的点
        vector<Point> all_points;
        for (const auto& lightBar : lightBars)
        {
            Point2f vertices[4];
            lightBar.points(vertices);
            for (int j = 0; j < 4; j++)
            {
                all_points.push_back(vertices[j]);
            }
        }

        // 绘制包围所有灯条的矩形
        if (!all_points.empty())
        {
            Rect bounding_rect = boundingRect(all_points);
            rectangle(result_image, bounding_rect, Scalar(255, 0, 0), 9);
            
            // 显示灯条数量
            string count_text = "Light Bars: " + to_string(lightBars.size());
            putText(result_image, count_text, Point(10, 30), 
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        }

        imshow("Video Feed", frame);
        imshow("Contours", result_image);
        imshow("Binary Image", binaryImage);
        
        if (waitKey(delay) >= 0)
            stop = true;
    }

    capture.release();
    return 0;
}