import argparse

import cv2
import numpy as np
import re

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, VisImage


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="For single image demo on Detectron2")
    parser.add_argument(
        "--base_model",
        default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        help="The default Base model to use for the traning",
    )

    parser.add_argument(
        "--images",
        nargs="+",
        help="A list of space separated image files"
        "Results will be saved next to the original images with "
        "'_processed_' appended to file name.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = parse_argument()

    cfg: CfgNode = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.base_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.base_model)
    predictor: DefaultPredictor = DefaultPredictor(cfg)

    img: str
    for img in args.images:
        img: np.ndarray = cv2.imread(img)

        output: Instances = predictor(img)["instances"]
        v = Visualizer(
            img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0
        )
        result: VisImage = v.draw_instance_predictions(output.to("cpu"))
        processed_img: np.ndarray = result.get_image()[:, :, ::-1]

        out_file_name: str = re.search(r"(.*)\.", img).group(0)[:-1]
        out_file_name += "_post_detectron2.png"

        cv2.imwrite(out_file_name, processed_img)
