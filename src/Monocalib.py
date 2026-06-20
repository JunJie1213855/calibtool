import cv2
import numpy as np
from .utiles import parse_yaml,getCorner,getworldcornerpoints,visulizationCorner,load_corner_points_from_csv,load_world_points_from_csv,save_corners_points,save_world_points,CalibrationConfig
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

class MonocularCameraCalibration:
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
        world_points, image_points = self.__load_points_from_csv()

        # 标定
        if self.args.camera_sensor_type == "Pinhole" or self.args.camera_sensor_type == "Fisheye":
            retval, cameraMatrix, distCoeffs, rvecs, tvecs = self.__calibAlgorithm(world_points, image_points)
        elif self.args.camera_sensor_type == "Omnidir":
            retval, cameraMatrix, xi , distCoeffs , rvecs, tvecs , idx = self.__calibAlgorithm(world_points, image_points)
            self.xi = xi

        logger.info("标定完成")
        logger.info(f"重投影误差 e : {retval:.6f}")
        logger.info(f"相机内参矩阵 K : \n{cameraMatrix}")
        logger.info(f"畸变系数 D :\n{distCoeffs}")

        # 保存
        logger.info(f"开始保存,保存目录为: {self.args.output_dir}")
        os.makedirs(self.args.output_dir,exist_ok=True)
        self.__save_intrinsics(cameraMatrix,distCoeffs)
        if self.args.camera_sensor_type == "Omnidir":
            omnidir_vaildpaths = []
            for index in idx[0]:
                omnidir_vaildpaths.append(self.valid_paths[int(index)])
            self.__save_extrinsics(rvecs,tvecs,omnidir_vaildpaths)

        if self.args.camera_sensor_type == "Pinhole" or self.args.camera_sensor_type == "Fisheye":
            self.__save_extrinsics(rvecs,tvecs,self.valid_paths)

    def __calibrate(self):
        self.valid_paths = []
        objpoints = []
        imgpoints = []

        # 获得单个世界坐标点集
        objp = getworldcornerpoints(self.args.board_size,self.args.square_size)
        logger.info("开始查找角点!")
        for path in tqdm(self.image_paths, desc="Processing images"):
            image = cv2.imread(path)
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = image
            
            ret, corners = getCorner(gray, self.args.board_size, self.args.radius_size, self.args.board_type, self.criteria)

            if not ret:
                message = f"没有找到角点,图像路径为: {path}"
                logger.info(message)
                tqdm.write(message)
                continue
            message = f"找到角点了,图像路径为: {path}"
            logger.info(message)
            tqdm.write(message)
            self.valid_paths.append(path)
            objpoints.append(objp)
            imgpoints.append(corners)
            visulizationCorner(image,corners,self.args.board_size,False) # 可视化
        logger.info("开始标定!")
        # 标定
        if self.args.camera_sensor_type == "Pinhole" or self.args.camera_sensor_type == "Fisheye":
            retval, cameraMatrix, distCoeffs, rvecs, tvecs = self.__calibAlgorithm(objpoints, imgpoints)
        elif self.args.camera_sensor_type == "Omnidir":
            retval, cameraMatrix, xi , distCoeffs , rvecs, tvecs , idx = self.__calibAlgorithm(objpoints, imgpoints)
            self.xi = xi

        logger.info("标定完成")
        logger.info(f"重投影误差 e : {retval:.6f}")
        logger.info(f"相机内参矩阵 K : \n{cameraMatrix}")
        logger.info(f"畸变系数 D :\n{distCoeffs}")

        # 保存
        logger.info(f"开始保存,保存目录为: {self.args.output_dir}")
        os.makedirs(self.args.output_dir,exist_ok=True)
        self.__save_intrinsics(cameraMatrix,distCoeffs)
        if self.args.camera_sensor_type == "Omnidir":
            omnidir_vaildpaths = []
            for index in idx[0]:
                omnidir_vaildpaths.append(self.valid_paths[int(index)])
            self.__save_extrinsics(rvecs,tvecs,omnidir_vaildpaths)

        if self.args.camera_sensor_type == "Pinhole" or self.args.camera_sensor_type == "Fisheye":
            self.__save_extrinsics(rvecs,tvecs,self.valid_paths)
        self.__save_corner_points_to_csv(imgpoints,objpoints)
    
    
    
    def __getdata(self):
        image_names = os.listdir(self.args.root_dir)
        self.image_paths = [os.path.join(self.args.root_dir, name) for name in image_names]
    
    def __calibAlgorithm(self, objpoints, imgpoints):
        if self.args.camera_sensor_type == "Pinhole":
            logger.info("Pinhole calibration!")
            retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                                                 imgpoints,
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
            self.args.flag += cv2.fisheye.CALIB_FIX_SKEW
            retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.fisheye.calibrate(objpoints,
                                                                                    imgpoints,
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
            retval, cameraMatrix, xi, distCoeffs, rvecs, tvecs, idx = cv2.omnidir.calibrate(objpoints,
                                                                                                imgpoints,
                                                                                                self.img_size,
                                                                                                K=None,
                                                                                                xi=None,
                                                                                                D=None,
                                                                                                flags=self.args.flag,
                                                                                                criteria=self.criteria
                                                                                                )
            return retval, cameraMatrix, xi , distCoeffs , rvecs, tvecs , idx
        
        return retval, cameraMatrix, distCoeffs, rvecs, tvecs
    
    def __imgsize(self):
        img = cv2.imread(self.image_paths[0])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h,w = gray.shape[:2]
        return (w,h)
    
    def __save_extrinsics(self, rvecs, tvecs, valid_paths):
        save_path = Path(self.args.output_dir) / "pose.txt"  # 可改为 self.args.extrinsic_save，如果你想分开存

        with open(save_path, 'w', encoding='utf-8') as f:
            for path, rvec, tvec in zip(valid_paths, rvecs, tvecs):
                name = os.path.basename(path)
                r_str = ' '.join([f'{v:.6f}' for v in rvec.flatten()])
                t_str = ' '.join([f'{v:.6f}' for v in tvec.flatten()])
                f.write(f"{name} {r_str} {t_str}\n")
        logger.info(f"每张图像的旋转/平移向量已保存到 {save_path}")
    
    def __save_corner_points_to_csv(self, imgpoints, objpoints):
        """
        将检测到的角点及其对应的世界坐标保存为CSV格式
        每行一个角点: 行索引,列索引,世界X坐标,世界Y坐标,世界Z坐标,图像x坐标,图像y坐标
        为每张图像创建一个CSV文件
        """
        # 创建输出目录
        # 角点
        corner_dir = os.path.join(self.args.output_dir, "corners")
        os.makedirs(corner_dir, exist_ok=True)
        save_corners_points(imgpoints,self.valid_paths,corner_dir,self.args.board_size)
        logger.info(f"角点坐标已保存到 {corner_dir} 目录")
        # 世界点
        world_coords_path = os.path.join(self.args.output_dir, "world_coordinates.csv")
        save_world_points(objpoints,world_coords_path,self.args.board_size)
        logger.info(f"世界坐标已保存到 {world_coords_path} 文件里面")


    def __save_intrinsics(self,K,D):
        save_path = str(Path(self.args.output_dir) / "param.yaml")  # 从配置中获取保存路径
        fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)

        fs.write("Camera_SensorType", self.args.camera_sensor_type)
        fs.write("Camera_NumType", self.args.camera_num_type)
        fs.write("K_l", K)
        fs.write("D_l", D)
        fs.write("height", self.img_size[1])
        fs.write("width", self.img_size[0])
        if self.args.camera_sensor_type == "Omnidir":
            fs.write("xi_l",self.xi)
        fs.release()
        logger.info(f"标定结果已保存到 {save_path}")

    def __load_points_from_csv(self):
        """从CSV文件加载角点数据和世界点数据"""
        if not self.args.use_csv_data:
            return [], []
        logger.info("从CSV文件加载角点数据")
        
        # 加载图像角点
        image_points,self.valid_paths = load_corner_points_from_csv(image_dir=self.args.root_dir,csv_dir=Path(self.args.image_points_dir) / "corners")
        
        # 加载世界点
        world_points = load_world_points_from_csv(self.args.world_points_file,len(image_points))
        return world_points, image_points