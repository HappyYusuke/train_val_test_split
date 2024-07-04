import os
import glob
import random
import shutil
import cupy as cp
from tqdm import tqdm
from data_augmentator import Augmentator


# 読み込む画像のディレクトリまでのパス
IMAGES_PATH = "/home/demulab/follow_me2024_dataset1/images"
# ラベルのディレクトリまでのパス
LABELS_PATH = "/home/demulab/follow_me2024_dataset1/labels"
# 保存先のディレクトリ名
save_dir = "train_val_test"


# train : val : test = 16 : 4 : 5
rate = {'train': 16, 
        'val': 4, 
        'test': 5}
total = sum(rate.values())

a = Augmentator(images_path=IMAGES_PATH, labels_path=LABELS_PATH)

# 画像とラベルの全パスを取得
image_paths = cp.array_str(a.get_images_path())
label_paths = cp.array_str(a.get_labels_path())
# データ数を格納
data_num = len(label_paths)
# 拡張子を取得
_, image_dot = a.get_file_name(image_paths[0])
_, label_dot = a.get_file_name(label_paths[0])

# ディレクトリがない場合は自動で作成する関数
def dir_check(path):
    if not os.path.exists(path):
        os.makedirs(path)
# 保存先のパスを作成
train_img_path = save_dir + "/images/train/"
train_label_path = save_dir + "/labels/train/"
val_img_path = save_dir + "/images/val/"
val_label_path = save_dir + "/labels/val/"
test_img_path = save_dir + "/images/test/"
test_label_path = save_dir + "/labels/test/"
# 保存先の有無を確認
path_list = [
        train_img_path, train_label_path, 
        val_img_path, val_label_path, 
        test_img_path, test_label_path
        ]
for path in path_list:
    dir_check(path=path)


print("Train data")
train_cnt = 1
for i in tqdm(range(int(len(label_paths) * (rate['train']/total)))):
    # ランダムにファイル名取得
    index = random.randint(0, data_num)
    file_name = os.path.splitext(os.path.basename(label_paths[index]))[0]
    # 取得したファイル名はリストから削除する
    image_paths = cp.array_str([path for path in image_paths if not file_name+image_dot in path])
    label_paths = cp.array_str([path for path in label_paths if not file_name+label_dot in path])
    # コピー元とコピー先のパスを作成する
    img_copy_from = IMAGES_PATH + "/" + file_name + image_dot
    img_copy_to = train_img_path + file_name + image_dot
    label_copy_from = LABELS_PATH + "/" + file_name + label_dot
    label_copy_to = train_label_path + file_name + label_dot
    # コピー
    shutil.copy(img_copy_from, img_copy_to)
    shutil.copy(label_copy_from, label_copy_to)

    data_num -= 1
    train_cnt += 1

print("Val data")
val_cnt = 1
for i in tqdm(range(int(len(label_paths) * (rate['val']/total)))):
    # ランダムにファイル名取得
    index = random.randint(0, data_num)
    file_name = os.path.splitext(os.path.basename(label_paths[index]))[0]
    # 取得したファイル名はリストから削除する
    image_paths = cp.array_str([path for path in image_paths if not file_name+image_dot in path])
    label_paths = cp.array_str([path for path in label_paths if not file_name+label_dot in path])
    # コピー元とコピー先のパスを作成する
    img_copy_from = IMAGES_PATH + "/" + file_name + image_dot
    img_copy_to = val_img_path + file_name + image_dot
    label_copy_from = LABELS_PATH + "/" + file_name + label_dot
    label_copy_to = val_label_path + file_name + label_dot
    # コピー
    shutil.copy(img_copy_from, img_copy_to)
    shutil.copy(label_copy_from, label_copy_to)

    data_num -= 1
    val_cnt += 1

print("Test data")
test_cnt = 1
for i in tqdm(range(int(len(label_paths) * (rate['test']/total)))):
    # ランダムにファイル名取得
    index = random.randint(0, data_num)
    file_name = os.path.splitext(os.path.basename(label_paths[index]))[0]
    # 取得したファイル名はリストから削除する
    image_paths = cp.array_str([path for path in image_paths if not file_name+image_dot in path])
    label_paths = cp.array_str([path for path in label_paths if not file_name+label_dot in path])
    # コピー元とコピー先のパスを作成する
    img_copy_from = IMAGES_PATH + "/" + file_name + image_dot
    img_copy_to = test_img_path + file_name + image_dot
    label_copy_from = LABELS_PATH + "/" + file_name + label_dot
    label_copy_to = test_label_path + file_name + label_dot
    # コピー
    shutil.copy(img_copy_from, img_copy_to)
    shutil.copy(label_copy_from, label_copy_to)

    data_num -= 1
    test_cnt += 1

print("All Completed!!!")
print()  # 改行
print(f"Train data: {train_cnt}")
print(f"Val data: {val_cnt}")
print(f"Test data: {test_cnt}")
print(f"Total: {train_cnt + val_cnt + test_cnt}")
