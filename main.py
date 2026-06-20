from src import MonocularCameraCalibration,StereoCameraCalibration,parse_yaml
import argparse

def main(args):
    config = parse_yaml(args.config)
    if config.camera_num_type == "Monocular" :
        calib = MonocularCameraCalibration(config)
        calib.calibrate()
    elif config.camera_num_type == "Stereo" :
        calib = StereoCameraCalibration(config)
        calib.calibrate()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config","-c",default="config/configstereo.yaml")
    args = parser.parse_args()
    main(args)