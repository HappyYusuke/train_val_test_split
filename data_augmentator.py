import os
import glob
import argparse
from tqdm import tqdm
# データ拡張系
import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image, ImageReadMode, write_jpeg


# 保存するファイル名
SAVE_NAME = "follow_me2024"
# 読み込むフォルダまでのパス
IMAGES_PATH = "/home/demulab/follow_me_dataset_origin/images"
LABELS_PATH = "/home/demulab/follow_me_dataset_origin/labels"
# GPUの設定
DEVICE = "cuda:0"
# 何回データ拡張の処理をループするか
LOOP_NUM = 14
# データ拡張の設定
TRANSFORMS = [
        T.Compose([
            T.ToImage(),

            
            # 切り取って指定されたサイズに変更する
            T.RandomResizedCrop(size=(700, 700), antialias=True),
            # 水平に反転
            T.RandomHorizontalFlip(p=0.5),
            # 鮮鋭化
            T.RandomAdjustSharpness(sharpness_factor=0 ,p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=3, p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=5, p=0.2),
            # アフィン変換
            T.RandomAffine(degrees=[-10, 10], translate=(0.2, 0.2), scale=(0.7, 1.5)),

            T.ToDtype(torch.uint8, scale=True)
            ]),

        T.Compose([
            T.ToImage(),

            # 射影変換(pは確率)
            T.RandomPerspective(p=0.3),
            # 鮮鋭化
            T.RandomAdjustSharpness(sharpness_factor=0 ,p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=3, p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=5, p=0.2),
            # 水平に反転
            T.RandomHorizontalFlip(p=0.5),
            
            T.ToDtype(torch.uint8, scale=True)
            ]),

        T.Compose([
            T.ToImage(),

            # 回転
            T.RandomRotation(degrees=20),
            # アフィン変換
            T.RandomAffine(degrees=[-10, 10], translate=(0.2, 0.2), scale=(0.7, 1.5)),
            # 水平に反転
            T.RandomHorizontalFlip(p=0.5),

            T.ToDtype(torch.uint8, scale=True)
            ]),
        ]


class Augmentator():
    def __init__(self, transforms=[], save_name="", images_path="", labels_path="", loop_num=1, device="cuda:0"):
        # 引数
        self.transforms = transforms
        self.save_name = save_name
        self.images_path = images_path
        self.labels_path = labels_path
        self.loop_num = loop_num
        self.device = device
        # 値
        self.count = 0
        self.imgs_num = 0
        self.save_dir = "results"
        self.images_savepath = self.save_dir + "/images"
        self.labels_savepath = self.save_dir + "/labels"
        self.bboximages_savepath = self.save_dir + "/bbox_images"
        self.save_fullpath = os.path.dirname(__file__) + "/" + self.save_dir

    def cuda_check(self, device):
        if torch.cuda.is_available() and "cuda" in device:
            d_type = device
            device_name = torch.cuda.get_device_name()
        else:
            d_type = device
            device_name = "cpu"
        print(f"Use device: {device_name}")

        return torch.device(d_type)

    def read_args(self):
        parser = argparse.ArgumentParser(
                prog="sample",
                usage="python3 data_augmentator.py <images_path> <labels_path>",
                description="Specify the paths to the images and labels directories you wish to expand.",
                epilog="end",
                add_help=True,
                )
        # 引数の設定
        parser.add_argument("images_path", type=str, help="Path to images directory.")
        parser.add_argument("labels_path", type=str, help="Path to labels directory.")
        # 引数の読み込み
        args = parser.parse_args()
        self.images_path = args.images_path
        self.labels_path = args.labels_path

    def get_images_path(self):
        return glob.glob(f"{self.images_path}/*")

    def get_labels_path(self):
        return glob.glob(f"{self.labels_path}/*")

    def get_file_name(self, path):
        basename = os.path.basename(path)
        file_name, extension = os.path.splitext(basename)
        return file_name, extension

    def dir_check(self):
        dirs = [self.save_dir, self.images_savepath, self.labels_savepath, self.bboximages_savepath]
        # ディレクトリがなければ作成する
        for path in dirs:
            if not os.path.exists(path):
                os.makedirs(path)

    def txt_to_box(self, txt_data, del_class=True):
        boxes = []
        # 情報をfloatに変換する
        for txt in txt_data.split('\n'):
            boxes.append([float(i) for i in txt.split(' ') if i != ''])
        # for文が一回多いので空のリストを削除
        for index in range(len(boxes)):
            if not boxes[index]:
                del boxes[index]
        # 0番目はクラス番号なので削除
        if del_class:
            for index in range(len(boxes)):
                del boxes[index][0]
        
        return boxes

    def yolobox_to_xyxy(self, boxes, img_size=(700, 700)):
        # 各座標をピクセル値に変換する
        px_list = []
        for value in boxes:
            bbox_px = list(map(lambda x: x*img_size[0], value))
            px_list.append(bbox_px)
        # xmin, ymin, xmax, ymaxに変換する
        xyxy_boxes = []
        for cxcywh_px in px_list:
            cx = cxcywh_px[0]
            cy = cxcywh_px[1]
            w = cxcywh_px[2]
            h = cxcywh_px[3]

            xmin = cx - (w/2)
            ymin = cy - (h/2)
            xmax = cx + (w/2)
            ymax = cy + (h/2)

            xyxy_boxes.append([xmin, ymin, xmax, ymax])

        return xyxy_boxes

    def xyxy_to_yolobox(self, boxes, img_size=(700, 700)):
        # yoloのフォーマットに変換する
        yolo_boxes = []
        for xyxy in boxes:
            xmin, ymin, xmax, ymax = list(map(lambda value1: float(value1), xyxy))
            # CXCYWHに変換する
            w = xmax - xmin
            h = ymax - ymin
            cx = xmax - (w/2)
            cy = ymax - (h/2)
            cxcywh_box = [cx, cy, w, h]
            # 画像サイズで割る
            yolo_box = list(map(lambda value2: round(float(value2)/img_size[0], 6), cxcywh_box)) 
            yolo_boxes.append(yolo_box)

        return yolo_boxes

    def good_data(self, boxes):
        good_flg = True

        for box in boxes:
            # 各値が末端にあるか
            for num in box:
                num = float(num)
                if num == 0.0 or num == 1.0:
                    good_flg = False
                    break
            # bboxの各辺が極端に小さくないか
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            if width <= 20 or height <= 20:
                good_flg = False
                break
            # 悪いデータだったら即座にbreak
            if not good_flg:
                break
        
        return good_flg

    def transforms_scheduler(self, loop_num):
        index_num = len(self.transforms)
        main_loop_num = int(loop_num/index_num)
        sub_loop_num = index_num
        return main_loop_num, sub_loop_num

    def interrupted(self):
        print()  # 改行
        print("Interrupted!!!")
        print(f"{self.count+1} images were augmentationed!")
        print(f"Save to {self.save_fullpath}")

    def augmentation(self, transforms_index=0):
        empty_files = []
        # transformsの表示
        print(self.transforms[transforms_index])
        # GPUの確認
        device = self.cuda_check(self.device)
        # 画像とラベルの全パスを取得
        image_paths = self.get_images_path()
        label_paths = self.get_labels_path()
        # 各ファイルの拡張子を取得
        _, image_dot = self.get_file_name(image_paths[0])
        _, label_dot = self.get_file_name(label_paths[0])
        # 保存先のディレクトリを確認
        self.dir_check()
        
        # 全ファイルを拡張する
        for img_path in tqdm(image_paths):
            # ファイル名のみ取得
            file_name, _ = self.get_file_name(img_path)
            # ファイル名と一致するラベルパスを取得する
            label_path = self.labels_path + '/' + file_name + label_dot
            
            # 画像ファイルの読み込み(チャネル数)
            img = read_image(img_path, ImageReadMode.RGB)
            img.to(device)
            img_size = T.functional.get_size(img)
            # ラベルの読み込み
            try:
                with open(label_path) as f:
                    txt_data = f.read()
            except FileNotFoundError:
                continue
            # ラベル情報がなければ最初からにする
            if not txt_data:
                empty_files.append(label_path)
                continue
            # クラス番号を保存
            class_nums = [int(data[0]) for data in self.txt_to_box(txt_data, del_class=False)]

            # YOLO形式からXYXY形式に変換
            boxes = self.txt_to_box(txt_data)
            xyxy_boxes = self.yolobox_to_xyxy(boxes=boxes, img_size=img_size)
            # torch用のbboxに変換
            torch_boxes = tv_tensors.BoundingBoxes(
                    xyxy_boxes,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=img.shape[-2:]
                    )
            
            # 拡張する
            img_ts, boxes_ts = self.transforms[transforms_index](img, torch_boxes)
            # 物体が画像になければ保存せず最初から
            if not self.good_data(boxes_ts):
                continue

            # 確認用の画像を生成
            bbox_img = draw_bounding_boxes(img_ts, boxes_ts, colors="red", width=3)
            
            # XYXYからYOLO形式に変換する
            yolo_boxes = self.xyxy_to_yolobox(boxes=boxes_ts, img_size=img_size)
            # 画像を保存
            write_jpeg(
                    input=img_ts,
                    filename=f"{self.images_savepath}/{self.save_name}_{self.count}{image_dot}")
            write_jpeg(
                    input=bbox_img,
                    filename=f"{self.bboximages_savepath}/{self.save_name}_{self.count}{image_dot}")
            # ラベルを保存
            with open(f"{self.labels_savepath}/{self.save_name}_{self.count}{label_dot}", mode='w') as f:
                for index in range(len(yolo_boxes)):
                    cx, cy, w, h = yolo_boxes[index]
                    bbox_string = f"{class_nums[index]} {cx} {cy} {w} {h}\n"
                    f.write(bbox_string)

            self.count += 1

        # ラベル情報がないファイルを出力
        if empty_files:
            print("The following files were empty.")
            for empty in empty_files:
                print(f" > {empty}")
        print()  # 改行

        return self.count

    def run(self):
        main_loop_num, sub_loop_num = self.transforms_scheduler(self.loop_num)
        count = 1
        for _ in range(main_loop_num):
            for transforms_index in range(sub_loop_num):
                print(f"Loop count: {count}/{self.loop_num}")
                images_num = self.augmentation(transforms_index=transforms_index)
                self.imgs_num += abs(images_num - self.imgs_num)
                count += 1
        
        print("All completed!")
        print(f"{self.imgs_num+1} images were augmentationed!")
        print(f"Save to {self.save_fullpath}")


if __name__ == '__main__':
    a = Augmentator(
            transforms=TRANSFORMS,
            save_name=SAVE_NAME,
            images_path=IMAGES_PATH,
            labels_path=LABELS_PATH,
            loop_num=LOOP_NUM,
            device=DEVICE
            )

    try:
        a.run()
    except KeyboardInterrupt:
        a.interrupted()
