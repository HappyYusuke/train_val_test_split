import os
import random
import shutil
import argparse
from tqdm import tqdm

# --- コマンドライン引数の設定 ---
parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets (Structure: split/train/images).")
parser.add_argument("-i", "--images", type=str, required=True, help="Path to source images directory.")
parser.add_argument("-l", "--labels", type=str, required=True, help="Path to source labels directory.")
parser.add_argument("-s", "--savename", type=str, default="split_result", help="Save directory name. (Default: split_result)")

# 分割比率の一括指定
parser.add_argument(
    "-r",
    "--ratio",
    type=int,
    nargs=3,
    default=[16, 4, 5],
    metavar=('TRAIN', 'VAL', 'TEST'),
    help="Ratio for train, val, and test data. (Default: 16 4 5)"
)

args = parser.parse_args()

# --- パスと設定 ---
IMAGES_PATH = args.images
LABELS_PATH = args.labels
save_dir = args.savename

# 比率計算
rate = {'train': args.ratio[0], 'val': args.ratio[1], 'test': args.ratio[2]}
total_rate = sum(rate.values())

if total_rate == 0:
    print("Error: Total ratio cannot be zero.")
    exit(1)

print(f"Splitting ratio (Train:Val:Test) = {rate['train']}:{rate['val']}:{rate['test']}")

# --- ファイル取得ロジック ---
print("Scanning files...")

def get_file_map(directory):
    file_map = {}
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return {}
    
    for filename in os.listdir(directory):
        if filename.startswith('.'): continue # 隠しファイルスキップ
        
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path):
            # 拡張子を除いた名前を取得
            stem = os.path.splitext(filename)[0]
            if stem in file_map:
                print(f"Warning: Duplicate filename stem found: {stem}")
            else:
                file_map[stem] = filename
    return file_map

image_map = get_file_map(IMAGES_PATH)
label_map = get_file_map(LABELS_PATH)

# ペアが存在するものだけ抽出
common_stems = list(set(image_map.keys()) & set(label_map.keys()))
common_stems.sort() # 再現性のためソート

data_num = len(common_stems)
print(f"Found {len(image_map)} images and {len(label_map)} labels.")
print(f"Valid pairs (Both exist): {data_num}")

if data_num == 0:
    print("Error: No matched image-label pairs found.")
    exit(1)

# --- ディレクトリ作成 (構成変更箇所) ---
def dir_check(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 保存先パスの定義を変更: save_dir/train/images/ の形にする
dirs = {
    'train': (
        os.path.join(save_dir, "train", "images"), 
        os.path.join(save_dir, "train", "labels")
    ),
    'val': (
        os.path.join(save_dir, "val", "images"),   
        os.path.join(save_dir, "val", "labels")
    ),
    'test': (
        os.path.join(save_dir, "test", "images"),  
        os.path.join(save_dir, "test", "labels")
    )
}

for img_dir, lbl_dir in dirs.values():
    dir_check(img_dir)
    dir_check(lbl_dir)

# --- シャッフルと分割 ---
random.shuffle(common_stems)

train_end = int(data_num * (rate['train'] / total_rate))
val_end = train_end + int(data_num * (rate['val'] / total_rate))

indices_split = {
    'train': common_stems[:train_end],
    'val':   common_stems[train_end:val_end],
    'test':  common_stems[val_end:]
}

# --- コピー実行 ---
total_cnt = 0

for phase in ['train', 'val', 'test']:
    stems = indices_split[phase]
    if not stems:
        continue
        
    print(f"Processing {phase} data...")
    save_img_dir, save_lbl_dir = dirs[phase]
    
    for stem in tqdm(stems):
        img_filename = image_map[stem]
        lbl_filename = label_map[stem]
        
        src_img = os.path.join(IMAGES_PATH, img_filename)
        src_lbl = os.path.join(LABELS_PATH, lbl_filename)
        
        dst_img = os.path.join(save_img_dir, img_filename)
        dst_lbl = os.path.join(save_lbl_dir, lbl_filename)
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)

    total_cnt += len(stems)

print("\nAll Completed!")
print(f"Output Directory Structure: {save_dir}/{{train,val,test}}/{{images,labels}}")
print(f"Total processed: {total_cnt}")
print(f"Train: {len(indices_split['train'])}, Val: {len(indices_split['val'])}, Test: {len(indices_split['test'])}")
