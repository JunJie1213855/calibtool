import cv2
import numpy as np
from .utiles import (
                    parse_yaml,
                    getworldcornerpoints,
                    load_corner_points_from_csv,
                    load_world_points_from_csv,
                    getStereoCorner,
                    visulizationCornerStereo,
                    save_corners_points,
                    save_world_points,
                    CalibrationConfig
                    )
import os
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,  # 可调为 DEBUG, WARNING, ERROR
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("calibration.log", mode='w', encoding='utf-8')  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)


def img2gray(image:np.ndarray):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image
    return gray


class StereoCameraCalibration:
    def __init__(self, args : CalibrationConfig):
        # 解析参数
        self.args = args
        self.criteria = (self.args.criteria[0],self.args.criteria[1],float(self.args.criteria[2]))
        self.__getdata()
        self.img_size = self.__imgsize()
    
    
    def calibrate(self):
        logger.info(f"摄像机数量 : {self.args.camera_num_type}")
        logger.info(f"摄像机类型 : {self.args.camera_sensor_type}")
        logger.info("开始标定!")
        if self.args.use_csv_data:
            self.__calibrate_csv()
            return self.args.use_csv_data
        self.__calibrate()
        return self.args.use_csv_data


    def __calibrate_csv(self):
        
        world_points, left_image_points, right_image_points = self.__load_points_from_csv()

        # 标定
        if self.args.camera_sensor_type == "Pinhole" or self.args.camera_sensor_type == "Fisheye":
            retval, K_l, D_l, K_r, D_r, R, t, left_rvecs, left_tvecs, right_rvecs, right_tvecs = self.__calibAlgorithm(world_points, left_image_points, right_image_points)
        elif self.args.camera_sensor_type == "Omnidir":
            retval, K_l, xi_l, D_l, K_r, xi_r, D_r, rvec, tvec, rvecsL, tvecsL, idx = self.__calibAlgorithm(world_points, left_image_points, right_image_points)
            self.xi_l = xi_l
            self.xi_r = xi_r
            R, _ = cv2.Rodrigues(rvec)
            t = tvec
        
        logger.info(f"双目重投影误差 error : {retval:.4f}")
        logger.info(f"左相机内参 K_l : \n{K_l}")
        logger.info(f"左相机内参 D_l : \n{D_l}")
        logger.info(f"右相机内参 K_r : \n{K_r}")
        logger.info(f"右相机内参 D_r : \n{D_r}")
        logger.info(f"旋转矩阵 R : \n{R}")
        logger.info(f"平移向量 t : \n{t}")
        
        # 保存
        logger.info(f"开始保存,保存目录为: {self.args.output_dir}")
        os.makedirs(self.args.output_dir,exist_ok=True)
        self.__save_intrinsics(K_l, D_l, K_r, D_r, R, t)
        self.__save_pair_record()


        if self.args.camera_sensor_type == "Omnidir":
            omnidir_vaildpaths = []
            for index in idx[0]:
                omnidir_vaildpaths.append(self.left_valid_paths[int(index)])
            self.__save_extrinsics(rvecsL,tvecsL,omnidir_vaildpaths,Path(self.args.output_dir) / "left_pose.txt")

        if self.args.camera_sensor_type == "Pinhole" or self.args.camera_sensor_type == "Fisheye":
            self.__save_extrinsics(left_rvecs,left_tvecs,self.left_valid_paths,Path(self.args.output_dir) / "left_pose.txt")
            self.__save_extrinsics(right_rvecs,right_tvecs,self.right_valid_paths,Path(self.args.output_dir) / "right_pose.txt")
        self.__save_corner_points_to_csv(left_image_points, right_image_points, world_points)
    






    def __calibrate(self):
        self.left_valid_paths = []
        self.right_valid_paths = []
        objpoints = []
        left_points = []
        right_points = []

        # 获得单个世界坐标点集
        objp = getworldcornerpoints(self.args.board_size,self.args.square_size)
        logger.info("开始查找角点!")
        index = 0
        for left_path,right_path in tqdm(zip(self.left_paths,self.right_paths), desc="Processing images"):
            left = cv2.imread(left_path)
            right = cv2.imread(right_path)
            left_gray = img2gray(left)
            right_gray = img2gray(right)
            ret,left_corners,right_corners = getStereoCorner(left_gray,right_gray,self.args.board_size,self.args.radius_size,self.args.board_type,self.criteria)
            if not ret:
                message = f"没有找到角点,图像路径为: {left_path}"
                logger.info(message)
                tqdm.write(message)
                continue # 下一帧
            message = f"找到了角点,图像路径为: {left_path}\n"
            logger.info(message)
            tqdm.write(message)
            # 有效图片
            self.left_valid_paths.append(left_path)
            self.right_valid_paths.append(right_path)
            # 标定点
            objpoints.append(objp)
            left_points.append(left_corners)
            right_points.append(right_corners)
            index = index + 1
            # 可视化
            visulizationCornerStereo(left,left_corners,right,right_corners,self.args.board_size,False) # 可视化
        objpoints = np.array(objpoints).reshape(len(objpoints),1,self.args.board_size[0] * self.args.board_size[1],3)
        left_points = np.array(left_points).reshape(len(left_points),1,self.args.board_size[0] * self.args.board_size[1],2)
        right_points = np.array(right_points).reshape(len(right_points),1,self.args.board_size[0] * self.args.board_size[1],2)
        # 标定
        logger.info("开始标定!")

        # 标定
        if self.args.camera_sensor_type == "Pinhole" or self.args.camera_sensor_type == "Fisheye":
            retval, K_l, D_l, K_r, D_r, R, t, left_rvecs, left_tvecs, right_rvecs, right_tvecs = self.__calibAlgorithm(objpoints, left_points, right_points)
        elif self.args.camera_sensor_type == "Omnidir":
            retval, K_l, xi_l, D_l, K_r, xi_r, D_r, rvec, tvec, rvecsL, tvecsL, idx = self.__calibAlgorithm(objpoints, left_points, right_points)
            self.xi_l = xi_l
            self.xi_r = xi_r
            R, _ = cv2.Rodrigues(rvec)
            t = tvec
        
        logger.info(f"双目重投影误差 error : {retval:.4f}")
        logger.info(f"左相机内参 K_l : \n{K_l}")
        logger.info(f"左相机内参 D_l : \n{D_l}")
        logger.info(f"右相机内参 K_r : \n{K_r}")
        logger.info(f"右相机内参 D_r : \n{D_r}")
        logger.info(f"旋转矩阵 R : \n{R}")
        logger.info(f"平移向量 t : \n{t}")
        
        # 保存
        logger.info(f"开始保存,保存目录为: {self.args.output_dir}")
        os.makedirs(self.args.output_dir,exist_ok=True)
        self.__save_intrinsics(K_l, D_l, K_r, D_r, R, t)
        self.__save_pair_record()


        if self.args.camera_sensor_type == "Omnidir":
            omnidir_vaildpaths = []
            for index in idx[0]:
                omnidir_vaildpaths.append(self.left_valid_paths[int(index)])
            self.__save_extrinsics(rvecsL,tvecsL,omnidir_vaildpaths,Path(self.args.output_dir) / "left_pose.txt")

        if self.args.camera_sensor_type == "Pinhole" or self.args.camera_sensor_type == "Fisheye":
            self.__save_extrinsics(left_rvecs,left_tvecs,self.left_valid_paths,Path(self.args.output_dir) / "left_pose.txt")
            self.__save_extrinsics(right_rvecs,right_tvecs,self.right_valid_paths,Path(self.args.output_dir) / "right_pose.txt")
        self.__save_corner_points_to_csv(left_points,right_points,objpoints)
    
    
    
    def __getdata(self):
        left_names = os.listdir(Path(self.args.root_dir) / "left")
        right_names = os.listdir(Path(self.args.root_dir) / "right")
        self.left_paths = [os.path.join(Path(self.args.root_dir) / "left", name) for name in left_names]
        self.right_paths = [os.path.join(Path(self.args.root_dir) / "right", name) for name in right_names]
    
    def __calibAlgorithm(self, objpoints, left_corers, right_corners):
        logger.info("初始化左右相机内参!")
        if self.args.camera_sensor_type == "Pinhole":
            logger.info("Pinhole calibration!")
            left_retval, K_l, D_l, left_rvecs, left_tvecs = cv2.calibrateCamera(objpoints,
                                                                                 left_corers,
                                                                                 self.img_size,
                                                                                 cameraMatrix=None,
                                                                                 distCoeffs=None,
                                                                                 rvecs=None,
                                                                                 tvecs=None,
                                                                                 flags=self.args.flag,
                                                                                 criteria=self.criteria
                                                                                 )
            right_retval, K_r, D_r, right_rvecs, right_tvecs = cv2.calibrateCamera(objpoints,
                                                                                 right_corners,
                                                                                 self.img_size,
                                                                                 cameraMatrix=None,
                                                                                 distCoeffs=None,
                                                                                 rvecs=None,
                                                                                 tvecs=None,
                                                                                 flags=self.args.flag,
                                                                                 criteria=self.criteria
                                                                                 )

        elif self.args.camera_sensor_type == "Fisheye":
            logger.info("Fisheye calibration!")
            left_retval, K_l, D_l, left_rvecs, left_tvecs = cv2.fisheye.calibrate(objpoints,
                                                                                 left_corers,
                                                                                 self.img_size,
                                                                                 K=None,
                                                                                 D=None,
                                                                                 rvecs=None,
                                                                                 tvecs=None,
                                                                                 flags=self.args.flag,
                                                                                 criteria=self.criteria
                                                                                 )
            right_retval, K_r, D_r, right_rvecs, right_tvecs = cv2.fisheye.calibrate(objpoints,
                                                                                 right_corners,
                                                                                 self.img_size,
                                                                                 K=None,
                                                                                 D=None,
                                                                                 rvecs=None,
                                                                                 tvecs=None,
                                                                                 flags=self.args.flag,
                                                                                 criteria=self.criteria
                                                                                 )

        elif self.args.camera_sensor_type == "Omnidir":
            logger.info("Omnidir calibration!")
            left_retval, K_l, xi_l, D_l, left_rvecs, left_tvecs, left_idx = cv2.omnidir.calibrate(objpoints,
                                                                                    left_corers,
                                                                                    self.img_size,
                                                                                    K=None,
                                                                                    xi=None,
                                                                                    D=None,
                                                                                    flags=self.args.flag,
                                                                                    criteria=self.criteria
                                                                                 )
            right_retval, K_r, xi_r, D_r, right_rvecs, right_tvecs, right_idx = cv2.omnidir.calibrate(objpoints,
                                                                                    right_corners,
                                                                                    self.img_size,
                                                                                    K=None,
                                                                                    xi=None,
                                                                                    D=None,
                                                                                    flags=self.args.flag,
                                                                                    criteria=self.criteria
                                                                                 )
        logger.info(f"左相机重投影误差 : {left_retval:.4f}")
        logger.info(f"右相机重投影误差 : {right_retval:.4f}")
        logger.info(f"初始化左相机内参 K_l : \n{K_l}")
        logger.info(f"初始化右相机内参 K_r: \n{K_r}")
        logger.info("开始计算双目相机的参数!")
        left_rvecs, left_tvecs, right_rvecs, right_tvecs = [], [], [], []
        if self.args.camera_sensor_type == "Pinhole":
            retval, K_l, D_l, K_r, D_r, R, t, E, F = cv2.stereoCalibrate(
                objpoints,
                left_corers,
                right_corners,
                cameraMatrix1=K_l,
                distCoeffs1=D_l,
                cameraMatrix2=K_r,
                distCoeffs2=D_r,
                imageSize=self.img_size,
                R=None,
                T=None,
                E=None,
                F=None,
                flags=self.args.flag,
                criteria=self.criteria
            )
            # 用新的内参计算外参
            for objp,left_corner,right_corner, in zip(objpoints,left_corers,right_corners):
                left_ret, left_rvec, left_tvec = cv2.solvePnP(objp,left_corner,K_l,D_l)
                right_ret, right_rvec, right_tvec = cv2.solvePnP(objp,right_corner,K_r,D_r)
                left_rvecs.append(left_rvec)
                left_tvecs.append(left_tvec)
                right_rvecs.append(right_rvec)
                right_tvecs.append(right_tvec)
        elif self.args.camera_sensor_type == "Fisheye":
            retval, K_l, D_l, K_r, D_r, R, t, rvecs, tvecs = cv2.fisheye.stereoCalibrate(
                                                        objpoints,
                                                        left_corers,
                                                        right_corners,
                                                        K1=K_l,
                                                        D1=D_l,
                                                        K2=K_r,
                                                        D2=D_r,
                                                        imageSize=self.img_size,
                                                        R =None,
                                                        T= None,
                                                        rvecs = None,
                                                        tvecs = None,
                                                        flags=self.args.flag,
                                                        criteria=self.criteria
                                                        )
            # 用新的内参计算外参
            for objp,left_corner,right_corner, in zip(objpoints,left_corers,right_corners):
                left_ret, left_rvec, left_tvec = cv2.fisheye.solvePnP(objp,left_corner,K_l,D_l)
                right_ret, right_rvec, right_tvec = cv2.fisheye.solvePnP(objp,right_corner,K_r,D_r)
                left_rvecs.append(left_rvec)
                left_tvecs.append(left_tvec)
                right_rvecs.append(right_rvec)
                right_tvecs.append(right_tvec)
        elif self.args.camera_sensor_type == "Omnidir":
            retval, _, _, _, K_l, xi_l, D_l, K_r, xi_r, D_r, rvec, tvec, rvecsL, tvecsL, idx = cv2.omnidir.stereoCalibrate(
                                                        objpoints,
                                                        left_corers,
                                                        right_corners,
                                                        imageSize1=self.img_size,
                                                        imageSize2=self.img_size,
                                                        K1=K_l,
                                                        xi1=xi_l,
                                                        D1=D_l,
                                                        K2=K_r,
                                                        xi2=xi_r,
                                                        D2=D_r,
                                                        flags=self.args.flag,
                                                        criteria=self.criteria
                                                        )
            return retval, K_l, xi_l, D_l, K_r, xi_r, D_r, rvec, tvec, rvecsL, tvecsL, idx
        return retval, K_l, D_l, K_r, D_r, R, t, left_rvecs, left_tvecs, right_rvecs, right_tvecs
    
    def __imgsize(self):
        img = cv2.imread(self.left_paths[0])
        gray = img2gray(img)
        h,w = gray.shape[:2]
        return (w,h)
    
    def __save_extrinsics(self, rvecs, tvecs, valid_paths, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            for path, rvec, tvec in zip(valid_paths, rvecs, tvecs):
                name = os.path.basename(path)
                r_str = ' '.join([f'{v:.6f}' for v in rvec.flatten()])
                t_str = ' '.join([f'{v:.6f}' for v in tvec.flatten()])
                f.write(f"{name} {r_str} {t_str}\n")
        logger.info(f"每张图像的旋转/平移向量已保存到 {save_path}")
    
    def __save_corner_points_to_csv(self, left_points, right_points, objpoints):
        """
        将检测到的角点及其对应的世界坐标保存为CSV格式
        每行一个角点: 行索引,列索引,世界X坐标,世界Y坐标,世界Z坐标,图像x坐标,图像y坐标
        为每张图像创建一个CSV文件
        """
        # 创建输出目录
        # 左角点
        left_corner_dir = os.path.join(self.args.output_dir, "left_corners")
        os.makedirs(left_corner_dir, exist_ok=True)
        save_corners_points(left_points,self.left_valid_paths,left_corner_dir,self.args.board_size)
        logger.info(f"左角点坐标已保存到 {left_corner_dir} 目录")
        
        # 右角点
        right_corner_dir = os.path.join(self.args.output_dir, "right_corners")
        os.makedirs(right_corner_dir, exist_ok=True)
        save_corners_points(right_points,self.right_valid_paths,right_corner_dir,self.args.board_size)
        logger.info(f"右角点坐标已保存到 {right_corner_dir} 目录")
        
        # 世界点
        world_coords_path = os.path.join(self.args.output_dir, "world_coordinates.csv")
        save_world_points(objpoints,world_coords_path,self.args.board_size)
        logger.info(f"世界坐标已保存到 {world_coords_path} 文件里面")

    def __save_intrinsics(self,K_l, D_l, K_r, D_r, R, t):
        save_path = Path(self.args.output_dir) / "param.yaml"  # 从配置中获取保存路径
        fs = cv2.FileStorage(str(save_path), cv2.FILE_STORAGE_WRITE)

        fs.write("Camera_SensorType", self.args.camera_sensor_type)
        fs.write("Camera_NumType", self.args.camera_num_type)

        fs.write("K_l", K_l)
        fs.write("D_l", D_l)
        fs.write("K_r", K_r)
        fs.write("D_r", D_r)
        if self.args.camera_sensor_type == "Omnidir":
            fs.write("xi_l",self.xi_l)
            fs.write("xi_r",self.xi_r)
        fs.write("R", R)
        fs.write("t", t)
        fs.write("height", self.img_size[1])
        fs.write("width", self.img_size[0])

        fs.release()
        logger.info(f"标定结果已保存到 {save_path}")

    def __save_pair_record(self):
        """双目: 记录左右目都成功检测到角点的图对. 按后缀数字匹配 (left01 ↔ right01).
        格式: 每行 'leftXX.jpg rightXX.jpg', 只列左右都有的对.
        """
        import re
        # 取文件名末尾的数字段 (left1.png -> 1, WIN_20230328_15_47_26_Pro.jpg -> 26)
        num_pat = re.compile(r"(\d+)\D*$")
        left_by_id, right_by_id = {}, {}
        for p in self.left_valid_paths:
            m = num_pat.search(os.path.basename(p))
            if m:
                left_by_id[m.group(1)] = os.path.basename(p)
        for p in self.right_valid_paths:
            m = num_pat.search(os.path.basename(p))
            if m:
                right_by_id[m.group(1)] = os.path.basename(p)
        common = sorted(set(left_by_id) & set(right_by_id))
        path = Path(self.args.output_dir) / "pairRecord.txt"
        with open(path, "w", encoding="utf-8") as f:
            for sid in common:
                f.write(f"{left_by_id[sid]} {right_by_id[sid]}\n")
        logger.info(f"双目配对记录已保存: {path} (共 {len(common)} 对)")



    def __load_points_from_csv(self):
        """从CSV文件加载角点数据和世界点数据"""
        if not self.args.use_csv_data:
            return [], [], []
        logger.info("从CSV文件加载角点数据")
        # 加载图像角点
        left_image_points,self.left_valid_paths = load_corner_points_from_csv(image_dir=os.path.join(self.args.root_dir,"left"),csv_dir=Path(self.args.image_points_dir) / "left_corners")
        right_image_points,self.right_valid_paths = load_corner_points_from_csv(image_dir=os.path.join(self.args.root_dir,"right"),csv_dir=Path(self.args.image_points_dir) / "right_corners")
           
        # 加载世界点
        world_points = load_world_points_from_csv(self.args.world_points_file,len(left_image_points))
        return world_points, left_image_points, right_image_points