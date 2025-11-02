#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
    VideoCapture cap("1351071190-1-160.mp4");
    
    if(!cap.isOpened()) {
        cout << "error, please check your file " << endl;
        return -1;
    }

    while(true) {
        Mat hsv, frame, blue_mask, blurred, result, final_mask;
        
        cap >> frame;
        if(frame.empty()) break;
        
        // 高斯模糊减少噪声
        GaussianBlur(frame, blurred, Size(3, 3), 0);
        //转换颜色空间
        cvtColor(blurred, hsv, COLOR_BGR2HSV);
        
        // 使用多个阈值范围来捕捉主要蓝色区域
        Mat blue_mask1, blue_mask2;
        inRange(hsv, Scalar(100, 150, 50), Scalar(120, 255, 255), blue_mask1);  // 核心蓝色
        inRange(hsv, Scalar(120, 100, 50), Scalar(140, 255, 255), blue_mask2);  // 浅蓝色（光晕）
        // 合并两个mask
        blue_mask = blue_mask1 | blue_mask2;
        
        // 形态学操作 - 先开运算去除小噪点，再闭运算连接断点
        Mat kernel_open = getStructuringElement(MORPH_RECT, Size(2, 2));
        Mat kernel_close = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(blue_mask, blue_mask, MORPH_OPEN, kernel_open);
        morphologyEx(blue_mask, blue_mask, MORPH_CLOSE, kernel_close);
        
        // 查找轮廓
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;//存放轮廓层级
        findContours(blue_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        //创建一个容器用来筛选灯条
        vector<RotatedRect> lightBars;
        
        // 初始化final_mask
        final_mask = Mat::zeros(blue_mask.size(), CV_8UC1);
        
        //开始循环遍历每一个灯条
        for(size_t i = 0; i < contours.size(); i++) {
 
            //利用旋转矩形拟合轮廓 
            RotatedRect rect = minAreaRect(contours[i]);

            //定义size2f结构体变量size用来获取旋转矩形的尺寸
            Size2f size = rect.size;

            //获取由旋转矩形拟合的轮廓的面积
            double area = size.height * size.width;

            //步骤1: 利用旋转矩形的面积先进行灯条的筛选
            if(area < 300 || area > 4000)
                continue;
            
            //步骤2: 用宽高比进行灯条的筛选
            float width = size.width;
            float height = size.height;
            float aspectRatio = max(width, height) / min(width, height);
            
            // 修正宽高比筛选条件
            if(aspectRatio < 2.0 || aspectRatio > 15.0)
                continue;

            //步骤3: 用角度进行灯条的筛选
            float angle = rect.angle;
            
            //角度调整：这里本来angle返回的值就是短边的角度，但是因为我们要获取的是长边的角度，所以要再旋转九十度
            if(width < height) {
                angle = 90 + angle;
            }

            //筛选接近竖直的灯条
            if(fabs(angle) < 60 || fabs(angle) > 120)
                continue;

            // 通过所有筛选条件，将灯条添加到容器
            lightBars.push_back(rect);
            
            // 在final_mask上绘制这个轮廓
            drawContours(final_mask, contours, i, Scalar(255), FILLED);
            
            // 绘制旋转矩形的四条边
            Point2f vertices[4];
            rect.points(vertices);
            for(int j = 0; j < 4; j++) {
                line(frame, vertices[j], vertices[(j+1)%4], Scalar(0, 255, 0), 2);
            }
        }
        
        // 通过mask提取目标区域
        bitwise_and(frame, frame, result, final_mask);

        // 窗口设置
        namedWindow("frame", WINDOW_NORMAL);
        namedWindow("result", WINDOW_NORMAL);
        namedWindow("blue_mask", WINDOW_NORMAL);
        resizeWindow("frame", 1500, 1300);
        resizeWindow("result", 1500, 1300);
        resizeWindow("blue_mask", 1500, 1300);
        
        imshow("frame", frame);
        imshow("result", result);
        imshow("blue_mask", blue_mask);
        
        char key = waitKey(30);
        if(key == 'q' || key == 27) break;
    }
    
    return 0;
}