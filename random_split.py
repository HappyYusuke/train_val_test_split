import os
import random
import shutil
import argparse
from tqdm import tqdm
# 'data_augmentator' は Augmentator クラスが定義されているファイル名と仮定
from data_augmentator import Augmentator 

# --- コマンドライン引数の設定 ---
parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets.")
parser.add_argument("-i", "--images", type=str, required=True, help="Your images dir path.")
parser.add_argument("-l", "--labels", type=str, required=True, help="Your labels dir path.")
parser.add_argument("-s", "--savename", type=str, default="split_result", help="Save dir name. (Default: split_result)")
parser.add_argument(
    "-r",
    "--ratio",
    type=int,
    nargs=3,  # 3つの引数を受け取る
    default=[16, 4, 5],  # デフォルト値
    metavar=('TRAIN', 'VAL', 'TEST'), # help表示用
    help="Ratio for train, val, and test data. (Default: 16 4 5)"
)

args = parser.parse_args()

# --- パスと比率の設定 ---
IMAGES_PATH = args.images
LABELS_PATH = args.labels
save_dir = args.savename

# 引数からrateディクショナリを作成
rate = {
    'train': args.ratio[0],
    'val': args.ratio[1],
    'test': args.ratio[2]
}
total_rate = sum(rate.values())

# 0除算を防ぐ
if total_rate == 0:
    print("Error: Total ratio cannot be zero.")
    exit(1)

# ユーザーに設定された比率を表示
print(f"Splitting dataset using ratio (Train:Val:Test) = {rate['train']}:{rate['val']}:{rate['test']}")
print(f"Source Images: {IMAGES_PATH}")
print(f"Source Labels: {LABELS_PATH}")
print(f"Destination: {save_dir}")

# --- Augmentatorの初期化とファイルリストの取得 ---
try:
    a = Augmentator(images_path=IMAGES_PATH, labels_path=LABELS_PATH)
    image_paths = a.get_images_path()
    label_paths = a.get_labels_path()
except Exception as e:
    print(f"Error initializing Augmentator or getting file paths: {e}")
    print("Please ensure 'data_augmentator.py' exists and paths are correct.")
    exit(1)

# データの整合性チェック
if len(image_paths) != len(label_paths):
    print(f"Warning: Image count ({len(image_paths)}) and label count ({len(label_paths)}) do not match.")

if not label_paths:
    print("Error: No labels found. Exiting.")
    exit(1)

# 拡張子の取得 (最初のファイルから取得)
try:
    _, image_dot = a.get_file_name(image_paths[0])
    _, label_dot = a.get_file_name(label_paths[0])
except IndexError:
    print("Error: Could not get file extensions, file lists might be empty.")
    exit(1)

# --- ディレクトリ作成 ---
def dir_check(path):
    """指定されたパスが存在しない場合、ディレクトリを作成します。"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# 保存先のパス定義
dirs = {
    'train': (os.path.join(save_dir, "images/train/"), os.path.join(save_dir, "labels/train/")),
    'val':   (os.path.join(save_dir, "images/val/"),   os.path.join(save_dir, "labels/val/")),
    'test':  (os.path.join(save_dir, "images/test/"),  os.path.join(save_dir, "labels/test/"))
}

# ディレクトリの作成
for img_dir, lbl_dir in dirs.values():
    dir_check(img_dir)
    dir_check(lbl_dir)

# --- 分割ロジック ---
# 全データのインデックスリストを作成 [0, 1, 2, ... n-1]
data_indices = list(range(len(label_paths)))
# ランダムにシャッフル
random.shuffle(data_indices)

# 分割位置の計算
total_files = len(data_indices)
train_end = int(total_files * (rate['train'] / total_rate))
# valの終点は、trainの数 + valの数
val_end = train_end + int(total_files * (rate['val'] / total_rate))

# インデックスをスライスで分割
indices_split = {
    'train': data_indices[:train_end],
    'val':   data_indices[train_end:val_end],
    'test':  data_indices[val_end:] # 残りすべてをtestに
}

# --- コピー処理実行 ---
total_cnt = 0
missing_files = 0

for phase in ['train', 'val', 'test']:
    print(f"--- Processing {phase.capitalize()} data ---")
    
    current_indices = indices_split[phase]
    if not current_indices:
        print(f"No files allocated for {phase} (Ratio might be too small or total files too few).")
        continue
        
    save_img_dir, save_lbl_dir = dirs[phase]
    
    for index in tqdm(current_indices, desc=f"Copying {phase} files"):
        try:
            # ラベルパスを取得
            src_lbl_path = label_paths[index]
            file_name = os.path.splitext(os.path.basename(src_lbl_path))[0]
            
            # 対応する画像パスを構築
            src_img_path = os.path.join(IMAGES_PATH, file_name + image_dot)
            
            # コピー先パス
            dst_img_path = os.path.join(save_img_dir, file_name + image_dot)
            dst_lbl_path = os.path.join(save_lbl_dir, file_name + label_dot)
            
            # ファイルが存在するか確認してからコピー
            if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
                shutil.copy(src_img_path, dst_img_path)
                shutil.copy(src_lbl_path, dst_lbl_path)
            else:
                if not os.path.exists(src_img_path):
                    print(f"\nWarning: Missing image file {src_img_path} (Label: {src_lbl_path})")
                if not os.path.exists(src_lbl_path):
                    print(f"\nWarning: Missing label file {src_lbl_path} (Image: {src_img_path})")
                missing_files += 1

        except Exception as e:
            print(f"\nError copying file at index {index}: {e}")

    print(f"{phase.capitalize()} count: {len(current_indices)}")
    total_cnt += len(current_indices)

# --- 最終結果の表示 ---
print("\n--- All Completed! ---")
print(f"Train data: {len(indices_split['train'])}")
print(f"Val data:   {len(indices_split['val'])}")
print(f"Test data:  {len(indices_split['test'])}")
print(f"Total processed: {total_cnt}")

if missing_files > 0:
    print(f"\nWarning: {missing_files} files were missing and skipped.")
