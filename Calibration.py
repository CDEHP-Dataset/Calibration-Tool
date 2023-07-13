def change_txt_to_array(data):
    point_list = []
    for x in data.replace("\n","").split("val")[1:]:
        temp = []
        x = x[11:]
        x = x.split()
        for i in range(0,len(x),2):
            temp.append(np.array([np.array([float(x[i]),float(x[i+1])])]).astype(np.float32))
        temp = np.array(temp)
        temp = temp.reshape((11,9,1,2)) #标定板网格形状
        temp_reshape = []
        for i in range(9):
            for j in range(11):
                temp_reshape.append(temp[j][i])
        temp_reshape = np.array(temp_reshape)
        point_list.append(temp_reshape)
    return point_list

import cv2
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
class shuangmu:
    def __init__(self):
        self.m1 = 0
        self.m2 = 0
        self.d1 = 0
        self.d2 = 0
        self.R = 0
        self.T = 0
stereo = shuangmu()

class StereoCalibration(object):
    def __init__(self):
        pass
    #标定图像
    def calibration_photo(self):
        #设置要标定的角点个数
        x_nums = 11                                                   #x方向上的角点个数
        y_nums = 9
        # 设置(生成)标定图在世界坐标中的坐标
        world_point = np.zeros((x_nums * y_nums,3),np.float32)            #生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素
        world_point[:,:2] = np.mgrid[:x_nums,:y_nums].T.reshape(-1, 2)    #mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行
                                                                            #.T矩阵的转置
                  
                #reshape()重新规划矩阵，但不改变矩阵元素
        #保存角点坐标
        
        
        #双目相机内外参数，用matlab获得
        mtxl = np.array([[613.9202270507812,0,429.2717590332031],[0,613.9828491210938,247.7591552734375],[0,0,1]])
        distl = np.array([0,0,0,0,0])
        mtxr = np.array([[1745.201222078004,0,608.7036499773853],[0,1745.160460801634,378.7029722868696],[0,0,1]])
        distr = np.array([-0.1078074878385094,-1.398091060704599, -0.001072933462189642, 0.001107858892628408, 8.071401568419741])
        
        # 读取matlab所得到的相机坐标
        with open(r'435i_reprojected.txt') as f:  
            data = f.read()
        image_positionl = change_txt_to_array(data)
        
        with open(r'celex5_reprojected.txt') as f:
            data = f.read()
        image_positionr = change_txt_to_array(data)
        stereo.m1 = mtxl
        stereo.m2 = mtxr
        
        stereo.d1 = distl
        stereo.d2 = distr
        
        world_position = []
        
        #标定所用的图像数量，全部使用则 range(len(image_positionr))
        for ii in range(50):
            world_position.append(world_point*38)
        
        ##双目标定
        self.stereo_calibrate( world_position ,image_positionl[:50], image_positionr[:50] , mtxl, distl, mtxr, distr, (1280,800))
            
    def stereo_calibrate( self ,  objpoints ,imgpoints_l , imgpoints_r , M1, d1, M2, d2, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        print("start wait!")
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
                                    objpoints, imgpoints_l,
                                    imgpoints_r, M1, d1, M2,
                                    d2, dims,
                                    criteria=stereocalib_criteria, flags=flags)
        print("done!!!")
        stereo.R = R
        
        stereo.T = T
        stereo.E = E
        stereo.F = F

#双目相机参数
class stereoCameral(object):
    def __init__(self):
        #左相机内参数
        self.cam_matrix_left = stereo.m1#np.array([[920.8802490234375,0,647.9076538085938],[0,920.9742431640625,371.63873291015625],[0,0,1]])
        #右相机内参数
        
        self.cam_matrix_right = stereo.m2#np.array([[1626.12175775792,0,657.208039931144],[0,1626.33420268197,381.838357114292],[0,0,1]])
        
        self.R = stereo.R#np.array([[0.999533238916758,-0.0299445563503751,-0.00605209431141959],[0.0301790369525112,0.998605509102330,0.0433158506680328],[0.00474658078979716,-0.0434782788925178,0.999043096785795]])
        #平移矩阵
        self.T = stereo.T#np.array([[44.9024612283526],[-160.371039865446],[796.884268762730]])
       
        
if __name__ == '__main__':
#     calibration_photo()
    biaoding = StereoCalibration()
    config = stereoCameral()
    np.save(r"m_l",config.cam_matrix_left)
    np.save(r"m_r",config.cam_matrix_right)
    np.save(r"R",config.R)
    np.save(r"T",config.T)

    biaoding.calibration_photo()