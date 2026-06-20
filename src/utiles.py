import cv2
import numpy as np
import yaml
from dataclasses import dataclass
from typing import Tuple
import yaml
import os
from pathlib import Path

@dataclass
class CalibrationConfig:
    root_dir: str
    camera_sensor_type: str
    camera_num_type: str
    square_size: float
    board_size: Tuple[int, int]
    board_type: str
    criteria: Tuple[int, int, float]
    flag: int
    omnidirflag : str
    radius_size: Tuple[int, int]
    output_dir : str
    use_csv_data: bool            # 是否使用CSV数据而非从图像检测角点
    image_points_dir: str # 图像角点CSV文件目录
    world_points_file: str # 世界坐标点CSV文件
    alpha: float

def parse_yaml(yaml_path: str) -> CalibrationConfig:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return CalibrationConfig(
        root_dir=config.get('root_dir'),
        camera_sensor_type=config.get('Camera_SensorType', 'Pinhole'),
        camera_num_type=config.get('Camera_NumType', 'Monocular'),
        square_size=config.get('square_size', 1.0),
        board_size=tuple(config.get('board_size', [7, 6])),
        board_type=config.get('board_type', 'Corner'),
        criteria=tuple(config.get('criteria', [3, 100, 1e-7])),
        flag=config.get('flag', 0),
        omnidirflag=config.get('omnidirflag',"RECTIFY_LONGLATI" ),
        radius_size=tuple(config.get('raduis_size', [5, 5])),
        output_dir=config.get('save_dir', "./output"),
        use_csv_data=config.get("use_csv_data",False),
        image_points_dir=config.get("image_points_dir",None),
        world_points_file=config.get("world_points_file",None),
        alpha=config.get("alpha",0.)
    )


def getCorner(img : np.ndarray, board_size, radius_size, board_type, criteria):
    if board_type=="Corner":
        "ret是是否找到角点的标志,corners就是角点集合"
        ret,corners=cv2.findChessboardCorners(img,board_size,cv2.CALIB_CB_ADAPTIVE_THRESH)
    elif board_type=="Circle":
        ret,corners=cv2.findCirclesGrid(img,board_size,cv2.CALIB_CB_ADAPTIVE_THRESH)
    
    if not ret:
        return ret,None
    
    if board_type == "Corner":
        "亚像素角点做细化"
        corners = cv2.cornerSubPix(img,corners,radius_size,(-1,-1),criteria)
    
    return ret, corners

def getStereoCorner(left : np.ndarray, right : np.ndarray, board_size, radius_size, board_type, criteria):
    if board_type=="Corner":
            "ret是是否找到角点的标志,corners就是角点集合"
            left_ret, left_corners = cv2.findChessboardCorners(left,board_size,None , cv2.CALIB_CB_ADAPTIVE_THRESH)
            right_ret, right_corners = cv2.findChessboardCorners(right, board_size, None ,cv2.CALIB_CB_ADAPTIVE_THRESH)
    elif board_type=="Circle":
            left_ret, left_corners = cv2.findCirclesGrid(left, board_size)
            right_ret, right_corners = cv2.findCirclesGrid(right, board_size)
    
    ret = (right_ret and left_ret)
    if not ret:
        return False, None, None

    if board_type=="Corner":
        "亚像素角点做细化"
        left_corners  = cv2.cornerSubPix(left,left_corners,radius_size,(-1,-1),criteria)
        right_corners = cv2.cornerSubPix(right, right_corners, radius_size, (-1, -1), criteria)
    return ret, left_corners, right_corners


def getworldcornerpoints(board_size, square_size):
    objp = np.zeros((1,board_size[0]*board_size[1],3),np.float32)
    "创建棋盘格世界坐标"
    objp[0,:,:2]  = np.mgrid[:board_size[0],:board_size[1]].T.reshape(-1,2) * square_size
    return objp


def visulizationCorner(img, corners, board_size, flag):
    cornerimg = cv2.drawChessboardCorners(img,board_size,corners,True)
    cv2.imshow("visual corners",cornerimg)
    if flag:
        cv2.waitKey()
        return flag
    cv2.waitKey(500)
    return flag


def visulizationCornerStereo(left, left_corners, right, right_corners,board_size, flag):
    left_cornerimg = cv2.drawChessboardCorners(left,board_size,left_corners,True)
    right_cornerimg = cv2.drawChessboardCorners(right,board_size,right_corners,True)
    cv2.imshow("left visual corners",left_cornerimg)
    cv2.imshow("right visual corners",right_cornerimg)
    if flag:
        cv2.waitKey()
        return flag
    cv2.waitKey(500)
    return flag




def load_corner_points_from_csv(image_dir, csv_dir, profix = "jpg"):
    # 加载图像点数据
    image_points = []
    valid_image_paths = []
    image_names = os.listdir(csv_dir)
    for name in image_names:
        csv_path = os.path.join(csv_dir, name)
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]  # 跳过表头
                    
            points = []
            for line in lines:
                values = line.strip().split(" ")
                x = float(values[0])
                y = float(values[1])
                points.append(np.array([x, y]))
                
        # 将点数据转换为OpenCV需要的格式
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        image_points.append(points_array)
        valid_image_paths.append(os.path.join(image_dir,f"{name.split('.')[0]}.{profix}"))
    return image_points,valid_image_paths


def load_corner_from_csv(csv_dir, profix = "jpg"):
    """
    通过csv文件名于角点进行对应
    corners_dict = {
        "xxx" : corner,
        ... 
    }
    """
    # 加载图像点数据
    image_names = os.listdir(csv_dir)
    corners_dict = {}
    for name in image_names:
        csv_path = os.path.join(csv_dir, name)
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]  # 跳过表头
                    
            points = []
            for line in lines:
                values = line.strip().split(" ")
                x = float(values[0])
                y = float(values[1])
                points.append(np.array([x, y]))
        corners_dict[f"{name.split('.')[0]}"] = np.array(points, dtype=np.float32)
    return corners_dict


def load_world_points_from_csv(csv_path, num_times):
    # 加载世界点数据
    world_points = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]  # 跳过表头
                
        points = []
        for line in lines:
            values = line.strip().split(" ")
            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
            points.append([x, y, z])
    p = np.array(points, dtype=np.float32).reshape(1,-1,3)
    for i in range(num_times):
        world_points.append(p)
    return world_points




def save_world_points(world_points, csv_path, board_size):
        for idx, objpoints in enumerate(world_points):
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("world_x world_y world_z\n")
                points = objpoints[0]  # 所有图像的世界坐标是一样的
                rows, cols = board_size
                
                for i in range(rows):
                    for j in range(cols):
                        point_idx = i * cols + j
                        X, Y, Z = points[point_idx]
                        f.write(f"{X:.6f} {Y:.6f} {Z:.6f}\n")


def save_corners_points(corner_points, valid_paths, corner_dir, board_size):
    """
    将检测到的角点及其对应的世界坐标保存为CSV格式
    每行一个角点: 行索引,列索引,世界X坐标,世界Y坐标,世界Z坐标,图像x坐标,图像y坐标
    为每张图像创建一个CSV文件
    """
    # 创建输出目录
    os.makedirs(corner_dir, exist_ok=True)
        
    for idx, (img_corners, img_path) in enumerate(zip(corner_points, valid_paths)):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        csv_path = os.path.join(corner_dir, f"{img_name}.csv")
            
        with open(csv_path, 'w', encoding='utf-8') as f:
            # 添加CSV头
            f.write("image_x image_y\n")
                
            # 写入每个角点
            img_corners = img_corners.reshape(-1, 2)  # 确保图像点格式正确
            rows, cols = board_size
                
            for i in range(rows):
                for j in range(cols):
                    # 计算在数组中的索引
                    point_idx = i * cols + j
                    if point_idx < len(img_corners):
                        # 图像坐标
                        x, y = img_corners[point_idx]
                        f.write(f"{x:.6f} {y:.6f}\n")


def load_pose_file(pose_path: Path) -> dict:
    """
    通过pose文件内部的图像名与位姿对应
    pose_dict = {
        "xxx" : (rvec, tvec),
        ... 
    }
    """
    poses_dict = {}
    with open(pose_path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            tokens = line.strip().split()
            name = tokens[0].split(".")[0]
            rvec = np.array([float(t) for t in tokens[1:4]], dtype=np.float64)
            tvec = np.array([float(t) for t in tokens[4:7]], dtype=np.float64)
            poses_dict[name] = (rvec, tvec)
    return poses_dict


def read_pair_record(pair_record_path):
    """读 pairRecord.txt. 自动识别单/双目格式:
    - 单目: 每行一个图名 → 返回 list[str]
    - 双目: 每行 "left right" → 返回 list[tuple[str, str]]
    文件不存在或为空 → 返回 [].
    """
    p = Path(pair_record_path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) >= 2:
                out.append((tokens[0], tokens[1]))
            else:
                out.append(tokens[0])
    return out
