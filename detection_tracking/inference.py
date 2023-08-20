import torch
import torchvision
import argparse
import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
from _utils import coco_category
import numpy as np
from sort import *
import numpy as np
from pathlib import Path


# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"
# sort_crab = Sort()


class Detector_Test:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vid_dir = args.vid_dir
        self.score_threshold = args.score_threshold
        self.sort_crab = Sort()

    def _load_pretrain_model(self) -> None:
        """Load the pretrain subject classification model"""

        # Load the pretrain subject predictor
        # TODO:deal with different model
        self.trained_model = torch.load(
            self.args.model_dir,
        )

    def __inference(self, frame, video_file, frame_id):
        # sort_crab = Sort()
        # print(self.trained_model)
        self.trained_model.eval()

        # img = Image.fromarray(frame.astype("uint8"), "RGB")
        # transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(frame)
        img = img.to(device)
        # print(img.shape)

        img = img.unsqueeze(0)
        prediction = self.trained_model(img)
        pred_score = list(prediction[0]["scores"].detach().cpu().numpy())
        coco_list = coco_category()

        # detection only
        if pred_score:
            pred_t = [pred_score.index(x) for x in pred_score][-1]

            if all(
                label == 1
                for label in list(prediction[0]["labels"].detach().cpu().numpy())
            ):
                pred_class = [
                    coco_list[i]
                    for i in list(prediction[0]["labels"].detach().cpu().numpy())
                ]
                pred_boxes = [
                    [(i[0], i[1]), (i[2], i[3])]
                    for i in list(prediction[0]["boxes"].detach().cpu().detach().numpy())
                ]

                pred_boxes = pred_boxes[: pred_t + 1]
                pred_class = pred_class[: pred_t + 1]

                for i in range(len(pred_boxes)):
                    if (pred_class[i]) == "crab" and pred_score[
                        i
                    ] > self.score_threshold:
                        cv2.rectangle(
                            frame,
                            (
                                int((pred_boxes[i][0])[0]),
                                int((pred_boxes[i][0])[1]),
                            ),
                            (
                                int((pred_boxes[i][1])[0]),
                                int((pred_boxes[i][1])[1]),
                            ),
                            (0, 0, 255),
                            2,
                        )

                        label_text = f"{pred_class[i]}: {pred_score[i]:.2f}"
                        cv2.putText(
                            frame,
                            label_text,
                            (
                                int((pred_boxes[i][0])[0]),
                                int((pred_boxes[i][0])[1]),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            thickness=2,
                        )

        return frame

        # sort
        # sort_crab = Sort()
        # if pred_score:
        #     pred_sort = []
        #     pred_t = [pred_score.index(x) for x in pred_score][-1]

        #     if all(
        #         label == 1
        #         for label in list(prediction[0]["labels"].detach().cpu().numpy())
        #     ):
        #         pred_class = [
        #             coco_list[i]
        #             for i in list(prediction[0]["labels"].detach().cpu().numpy())
        #         ]
        #         pred_boxes = [
        #             [(i[0], i[1]), (i[2], i[3])]
        #             for i in list(prediction[0]["boxes"].detach().cpu().detach().numpy())
        #         ]

        #         pred_boxes = pred_boxes[: pred_t + 1]
        #         pred_class = pred_class[: pred_t + 1]

        #         for pred_i in range(len(pred_boxes)):
        #             if (pred_class[pred_i]) == "crab" and pred_score[
        #                 pred_i
        #             ] > self.score_threshold:
        #                 bbox = np.asarray(pred_boxes[pred_i])
        #                 score = np.asarray(pred_score[pred_i])
        #                 pred_x = np.append(bbox, score)
        #                 pred_sort.append(pred_x)
        #         if pred_sort:
        #             pred_sort = np.asarray(pred_sort)
        #         else:
        #             pred_sort = np.empty((0, 5))
        #     else:
        #         pred_sort = np.empty((0, 5))        
        # else:
        #     pred_sort = np.empty((0, 5))
        # # print(pred_sort.shape)
        # sort_bbs_ids = self.sort_crab.update(pred_sort)

        # for sort_i in range(sort_bbs_ids.shape[0]):
        #     [x1, y1, x2, y2] = sort_bbs_ids[sort_i, 0:4]
        #     cv2.rectangle(
        #         frame,
        #         (int(x1), int(y1)),
        #         (int(x2), int(y2)),
        #         (0, 0, 255),
        #         2,
        #     )
        #     id_label = f"id : {sort_bbs_ids[sort_i][4]}"
        #     cv2.putText(
        #         frame,
        #         id_label,
        #         (int(x1), int(y1)),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 0, 255),
        #         thickness=2,
        #     )

        # return frame

    def _load_video(self) -> None:
        """Load images and annotation file for training"""

        print(self.vid_dir)
        try:
            video = cv2.VideoCapture(self.vid_dir)

            if not video.isOpened():
                raise Exception("Error opening video file")

            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_fps = video.get(cv2.CAP_PROP_FPS)

            output_file = "output_video.mp4"
            output_codec = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_file, output_codec, cap_fps, (frame_width, frame_height)
            )
            frame_id = 0

            video_file = (
                f"{Path(self.vid_dir).parent.stem}_" f"{Path(self.vid_dir).stem}_"
            )
            while video.isOpened():
                ret, frame = video.read()

                if not ret:
                    # Break the loop if no more frames to read
                    break

                # print(frame.shape)

                frame_out = self.__inference(frame, video_file, frame_id)
                frame_id += 1
                out.write(frame_out)
                # cv2.imshow("frame", frame_out)

            video.release()
            out.release()
            cv2.destroyAllWindows()

        except:
            print("Could not open video file")
            raise

    def inference_model(self) -> None:
        self._load_pretrain_model()
        # self.trained_model.eval()
        self._load_video()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="location of trained model",
    )
    parser.add_argument(
        "--vid_dir",
        type=str,
        required=True,
        help="location of images and coco annotation",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
        help="save video inference",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.getcwd(),
        help="location of output video",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="threshold for prediction score",
    )
    parser.add_argument(
        "--sort",
        type=bool,
        default=False,
        help="running sort as tracker",
    )

    args = parser.parse_args()
    inference = Detector_Test(args)
    inference.inference_model()
