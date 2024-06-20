import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import math

N_pix = 45

# データを読み込む
map_data = np.load('map_45.npy')

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

# ピークにIDを割り当てる
peak_ids = {}
if (N_pix % 2 == 0):
    center_id = (N_pix/2, N_pix/2)
else:
    center_id = ((N_pix-1)/2, (N_pix-1)/2)
peak_ids[tuple(center_peak)] = center_id

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

        if new_id in peak_ids:
            break

        peak_ids[tuple(next_peak)] = new_id
        current_id = new_id
        current_peak = next_peak

# 各方向に対してIDを割り当てる
print("left")
assign_id_in_direction(filtered_peaks, center_id, center_peak, 'left')
print("right")
assign_id_in_direction(filtered_peaks, center_id, center_peak, 'right')


# 上下左右の各ピークに対して同様に探索を繰り返す
for direction in ['up', 'down']:
    for current_peak, current_id in list(peak_ids.items()):
        if (N_pix % 2 == 0):
            if (current_id[1] == N_pix/2):
                assign_id_in_direction(filtered_peaks, current_id, np.array(current_peak), direction)
        else:
            if (current_id[1] == (N_pix-1)/2):
                assign_id_in_direction(filtered_peaks, current_id, np.array(current_peak), direction)

# IDを割り当てたピークを表示
plt.figure()
plt.imshow(map_data, cmap='viridis', origin='lower')
for (y, x), (id_x, id_y) in peak_ids.items():
    plt.plot(x, y, 'r,', markersize=10)
    plt.text(x, y, f'({id_x},{id_y})', color='white', fontsize=8, ha='center')
plt.title('Detected Peaks with IDs')
plt.colorbar(label='Intensity')
plt.show()

# CSVに出力する
# for peak_pos, peak_id in list(peak_ids.items()):

peak_ids
