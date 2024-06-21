#20240619
#by Gori
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import math

N_pix = 45
input_file_path = 'testdata/map_45.npy'
output_file_path = 'testdata/output.csv'

def coordinate_norm(value, min, original_size):
    value /= original_size / 2
    value += min

    return value

# データを読み込む
map_data = np.load(input_file_path).T
map_size_rows, map_size_cols = map_data.shape
print(map_size_cols)

# ガウシアンフィルタを適用して平滑化
smoothed_data = gaussian_filter(map_data, sigma=1)

# ピークを検出
initial_peaks = peak_local_max(smoothed_data, min_distance=7, threshold_abs=np.mean(smoothed_data))

# 格子パターンの検証
lattice_spacing = 5  # 予想される格子間隔
filtered_peaks = []

for peak in initial_peaks:
    y, x = peak
    if (y % lattice_spacing < 5 or y % lattice_spacing > lattice_spacing - 5) and \
       (x % lattice_spacing < 5 or x % lattice_spacing > lattice_spacing - 5):
        filtered_peaks.append(peak)

filtered_peaks = np.array(filtered_peaks)

# ピークを表示（IDを振る前）
plt.figure()
plt.imshow(map_data, cmap='viridis', origin='lower')
for peak in filtered_peaks:
    y, x = peak
    plt.plot(x, y, 'r.', markersize=10)
plt.title('Detected Peaks (Before Assigning IDs)')
plt.colorbar(label='Intensity')
plt.show()

# 最も中心に近いピークを見つける
center_point = np.array([map_data.shape[0] // 2, map_data.shape[1] // 2])
distances = np.linalg.norm(filtered_peaks - center_point, axis=1)
center_index = np.argmin(distances)
center_peak = filtered_peaks[center_index]

# ピークID格納変数の初期化
initial_pos = np.full(2, -1)
peak_ids = np.full((N_pix, N_pix, 2), initial_pos)
# for i in range(N_pix):
#     for j in range(N_pix):
#         peak_ids[i][j].append(initial_pos)

# ピークにIDを割り当てる
if (N_pix % 2 == 0):
    center_id = (N_pix/2, N_pix/2)
else:
    center_id = ((N_pix-1)/2, (N_pix-1)/2)
peak_ids[int(center_id[0])][int(center_id[1])] = center_peak

def assign_id_in_direction(peaks, start_id, start_peak, direction, max_dist=12, max_count=15, search_range=15, offset=5):
    current_id = start_id
    current_peak = start_peak

    while True:
        # 現在のピークの座標とID
        y, x = current_peak
        id_x, id_y = current_id

        # print(current_peak)

        # x方向の探索
        if direction == 'left':
            print(current_id)
            next_peaks = [peak for peak in peaks if peak[1] < x - offset and abs(peak[0] - y) < search_range]
        elif direction == 'right':
            next_peaks = [peak for peak in peaks if peak[1] > x + offset and abs(peak[0] - y) < search_range]
        elif direction == 'up':
            next_peaks = [peak for peak in peaks if peak[0] < y - offset and abs(peak[1] - x) < search_range]
        elif direction == 'down':
            next_peaks = [peak for peak in peaks if peak[0] > y + offset and abs(peak[1] - x) < search_range]

        # 次のピーク候補を取得
        if direction in ['left', 'right']:
            next_peaks = sorted(next_peaks, key=lambda p: abs(p[1] - x))[:max_count]
            # print(next_peaks)
        else:
            next_peaks = sorted(next_peaks, key=lambda p: abs(p[0] - y))[:max_count]
            # print(next_peaks)

        # 最も近いピークを選択
        if direction in ['left', 'right']:
            next_peaks = sorted(next_peaks, key=lambda p: math.sqrt((p[0] - y)**2 + (p[1] - x)**2))
        else:
            next_peaks = sorted(next_peaks, key=lambda p: math.sqrt((p[0] - y)**2 + (p[1] - x)**2))

        # 有効な次のピークがない場合、終了
        if not next_peaks:
            break

        next_peak = next_peaks[0]


        # 次のピークが有効範囲内かどうかをチェック
        if direction in ['left', 'right']:
            print(next_peak)
            if abs(next_peak[0] - y) > max_dist:
                print("Next peak is too far!!")
                break
        else:
            if abs(next_peak[1] - x) > max_dist:
                print("Next peak is too far!!")
                break

        # IDを割り当てる
        if direction == 'left':
            new_id = (id_x - 1, id_y)
        elif direction == 'right':
            new_id = (id_x + 1, id_y)
        elif direction == 'up':
            new_id = (id_x, id_y - 1)
        elif direction == 'down':
            new_id = (id_x, id_y + 1)

        # IDが範囲外の場合、終了
        if not (0 <= new_id[0] <= (N_pix - 1) and 0 <= new_id[1] <= (N_pix - 1)):
            break

        if peak_ids[int(new_id[0])][int(new_id[1])][0] != -1:
            break

        peak_ids[int(new_id[0])][int(new_id[1])] = next_peak
        current_id = new_id
        current_peak = next_peak

# 各方向に対してIDを割り当てる
print("left")
assign_id_in_direction(filtered_peaks, center_id, center_peak, 'left')
print("right")
assign_id_in_direction(filtered_peaks, center_id, center_peak, 'right')


# 上下左右の各ピークに対して同様に探索を繰り返す
for direction in ['up', 'down']:
    for i, a in enumerate(peak_ids):
        for j, current_peak in enumerate(a):
            current_id = [i, j]
            if (N_pix % 2 == 0):
                if (j == N_pix/2):
                    assign_id_in_direction(filtered_peaks, current_id, np.array(current_peak), direction)
            else:
                if (j == (N_pix-1)/2):
                    assign_id_in_direction(filtered_peaks, current_id, np.array(current_peak), direction)

# IDを割り当てたピークを表示
plt.figure()
plt.imshow(map_data, cmap='viridis', origin='lower')
for i, a in enumerate(peak_ids):
    for j, pos in enumerate(a):
        plt.plot(pos[1], pos[0], 'r,', markersize=10)
        plt.text(pos[1], pos[0], f'({i},{j})', color='white', fontsize=8, ha='center')
plt.title('Detected Peaks with IDs')
plt.colorbar(label='Intensity')
plt.show()

# CSVに出力する
miss_count = 0
with open(output_file_path, "w") as f:
    for i, a in enumerate(peak_ids):
        for j, pos in enumerate(a):
            if (i == 0 and j == 0):
                f.write("IDx,IDy,Posix,Posiy,accuracy\n")
            if (pos[0] == -1):
                f.write(f'{int(i)},{int(j)},{pos[1]},{pos[0]},miss\n')
                miss_count += 1
            else:
                f.write(f'{int(i)},{int(j)},{coordinate_norm(pos[1], -1, map_size_cols)},{coordinate_norm(pos[0], -1, map_size_cols)},\n')

print(miss_count)
peak_ids
