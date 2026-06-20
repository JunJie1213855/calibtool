"""
标定后验证入口
- 读 config.yaml
- 调用 PostCalibration 生成重投影统计 + 2D/3D 可视化
- 产物统一在 output/verify/

用法:
    python verify.py -c config/config.yaml
"""
import argparse
import logging
import sys

from src.postcalib import PostCalibration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(args):
    logger.info(f"加载配置: {args.config}")
    pc = PostCalibration(args.config)
    logger.info(f"开始验证 (模式: {pc.config.camera_num_type}, "
                f"传感器: {pc.config.camera_sensor_type})")
    pc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="标定后验证: 重投影误差统计 + 2D/3D 可视化")
    parser.add_argument("--config", "-c", default="config/config.yaml",
                        help="标定配置文件路径")
    main(parser.parse_args())
