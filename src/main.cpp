#include <epipolar_geometry.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <ctime>
clock_t startTime, endTime;

int main() {
    // 初始化测试
    std::cout << "Epipolar Geometry Lib Test" << std::endl;
    EpipolarGeometryClass EpipolarGeometry;

    // 构造 3D 点云
    std::vector<Eigen::Vector3d> pointCloud;
    for (int i = 1; i < 10; i++) {
        for (int j = 1; j < 10; j++) {
            for (int k = 1; k < 10; k++) {
                pointCloud.emplace_back(Eigen::Vector3d(i, j, k * 2.0));
            }
        }
    }

    // 定义两帧位姿
    Eigen::Matrix3d R_c0w = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_c0w = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R_c1w = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_c1w = Eigen::Vector3d(1, -3, 0);

    // 将 3D 点云通过两帧位姿映射到对应的归一化平面上，构造匹配点对
    std::vector<cv::Point2f> pixelPoints0, pixelPoints1;
    for (unsigned long i = 0; i < pointCloud.size(); i++) {
        Eigen::Vector3d tempP = R_c0w * pointCloud[i] + t_c0w;
        pixelPoints0.emplace_back(cv::Point2f(tempP(0, 0) / tempP(2, 0), tempP(1, 0) / tempP(2, 0)));
        tempP = R_c1w * pointCloud[i] + t_c1w;
        pixelPoints1.emplace_back(cv::Point2f(tempP(0, 0) / tempP(2, 0), tempP(1, 0) / tempP(2, 0)));
    }

    // 随机给匹配点增加 outliers
    std::vector<int> outliersIndex;
    for (unsigned int i = 0; i < pointCloud.size() / 100; i++) {
        int idx = rand() % pointCloud.size();
        pixelPoints1[idx].x += 5;
        outliersIndex.emplace_back(idx);
    }

    // 定义相机内参
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1 );
    Eigen::Matrix3d CameraMatrix;
    CameraMatrix << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    double fx = CameraMatrix(0, 0);
    double fy = CameraMatrix(1, 1);
    double cx = CameraMatrix(0, 2);
    double cy = CameraMatrix(1, 2);
    std::cout << std::endl;









    /*-----------------------------------------------------------------------------------------------------*/
    // 根据匹配点对，计算出本质矩阵
    std::vector<uchar> status;
    std::cout << "OpenCV compute Essential Matrix" << std::endl;
    cv::Mat E_cv = cv::findEssentialMat(pixelPoints0, pixelPoints1, K, cv::RANSAC, 0.99, 1.0, status);
    std::cout << E_cv << std::endl;
    // 根据匹配点对，计算出本质矩阵
    std::cout << "My Code compute Essential Matrix" << std::endl;
    double error;
    Eigen::Matrix3d E = EpipolarGeometry.FindEssentialMatrix(pixelPoints0, pixelPoints1, CameraMatrix, error);
    std::cout << E << std::endl;
    std::cout << std::endl;
    /*-----------------------------------------------------------------------------------------------------*/
    // 验证我编写的代码的对极约束
    double sum = 0.0;
    for (unsigned int i = 0; i < pixelPoints0.size(); i++) {
        Eigen::Vector3d p1 = Eigen::Vector3d((pixelPoints1[i].x - cx) / fx, (pixelPoints1[i].y - cy) / fy, 1);
        Eigen::Vector3d p0 = Eigen::Vector3d((pixelPoints0[i].x - cx) / fx, (pixelPoints0[i].y - cy) / fy, 1);
        double result = p1.transpose() * E * p0;
        sum += abs(result);
    }
    std::cout << "My epiplar constraint result average is " << sum / double(pixelPoints0.size()) << std::endl;

    // 验证 OpenCV 代码的对极约束
    E << E_cv.at<double>(0, 0), E_cv.at<double>(0, 1), E_cv.at<double>(0, 2),
         E_cv.at<double>(1, 0), E_cv.at<double>(1, 1), E_cv.at<double>(1, 2),
         E_cv.at<double>(2, 0), E_cv.at<double>(2, 1), E_cv.at<double>(2, 2);
    sum = 0.0;
    for (unsigned int i = 0; i < pixelPoints0.size(); i++) {
        Eigen::Vector3d p1 = Eigen::Vector3d((pixelPoints1[i].x - cx) / fx, (pixelPoints1[i].y - cy) / fy, 1);
        Eigen::Vector3d p0 = Eigen::Vector3d((pixelPoints0[i].x - cx) / fx, (pixelPoints0[i].y - cy) / fy, 1);
        double result = p1.transpose() * E * p0;
        sum += abs(result);
    }
    std::cout << "OpenCV epiplar constraint result average is " << sum / double(pixelPoints0.size()) << std::endl;
    std::cout << std::endl;










    /*-----------------------------------------------------------------------------------------------------*/
    // 从本质矩阵中恢复出 R 和 t
    std::cout << "My Code recover R and t result" << std::endl;
    status.clear();
    Eigen::Matrix3d R_c1c0;
    Eigen::Vector3d t_c1c0;
    startTime = clock();
    EpipolarGeometry.EstimateRotationAndTranslation(pixelPoints0, pixelPoints1, CameraMatrix, status, R_c1c0, t_c1c0);
    endTime = clock();
    std::cout << "Time cost " << (double)(endTime - startTime) / CLOCKS_PER_SEC << std::endl;
    std::cout << R_c1c0 << std::endl << t_c1c0 << std::endl;

    std::cout << "Opencv recover R and t result" << std::endl;
    status.clear();
    cv::Mat R_cv, t_cv;
    startTime = clock();
    E_cv = cv::findEssentialMat(pixelPoints0, pixelPoints1, K, cv::RANSAC, 0.99, 1.0, status);
    cv::recoverPose(E_cv, pixelPoints0, pixelPoints1, K, R_cv, t_cv);
    endTime = clock();
    std::cout << "Time cost " << (double)(endTime - startTime) / CLOCKS_PER_SEC << std::endl;
    std::cout << R_cv << std::endl << t_cv << std::endl;

    std::cout << "The real R_c1c0 and t_c1c0 is" << std::endl;
    std::cout << R_c1w * R_c0w.transpose() << std::endl;
    std::cout << t_c1w - t_c0w << std::endl;


    /*-----------------------------------------------------------------------------------------------------*/
    // 检查 status 是否符合预期
    status.clear();
    EpipolarGeometry.FindInliersWithEssentialMatrix(pixelPoints0, pixelPoints1, CameraMatrix, status);
    std::cout << "outliers detect result:" << std::endl;
    for (unsigned int i = 0; i < outliersIndex.size(); i++) {
        std::cout << (int)status[outliersIndex[i]];
    }
    std::cout << "\ninliers detect result:" << std::endl;
    for (unsigned int i = 0; i < status.size(); i++) {
        if (i % 100 == 0 && i != 0) {
            std::cout << std::endl;
        }
        std::cout << (int)status[i];
    }
    std::cout << std::endl;

    cv::waitKey();
    return 0;
}