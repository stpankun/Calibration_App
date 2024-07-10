#20240613
#by Gori
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Canvas, Scrollbar, Frame
import pandas as pd

def load_dat_image(dat_path, image_width, image_height, cmap='jet'):
    data = np.fromfile(dat_path, sep="\n", dtype=np.uint8)
    if data.size != image_width * image_height:
        raise ValueError(f"Data size {data.size} does not match expected size {image_width * image_height}")
    data = data.reshape((image_height, image_width)).T

    # Apply colormap
    norm = Normalize(vmin=0, vmax=255)
    cmap = plt.get_cmap(cmap)
    mapped_data = cmap(norm(data))
    mapped_data = (mapped_data[:, :, :3] * 255).astype(np.uint8)  # Discard the alpha channel and convert to uint8

    return Image.fromarray(mapped_data)

def normalize_coordinates(x, y, width, height):
    norm_x = (2 * x / width) - 1
    norm_y = (2 * y / height) - 1
    return norm_x, norm_y

def denormalize_coordinates(norm_x, norm_y, width, height):
    x = ((norm_x + 1) / 2) * width
    y = ((norm_y + 1) / 2) * height
    return x, y

# 画像サイズとファイルパスを定義
image_width = 1000 # bin数(基本は1000)
image_height = 1000
scale_factor = 0.5  # 画像をスケーリングする割合
scaled_width = int(image_width * scale_factor)
scaled_height = int(image_height * scale_factor)
cmap = 'jet'  # カラーマップの種類を指定

dat_path = 'testdata/output/map.dat'  # datファイルのパスを指定
csv_path = 'testdata/output/output.csv'  # ピーク位置のCSVファイルのパスを指定
# out_path = 'testdata/output/adjusted_peaks.csv'
out_path = 'testdata/output/output.csv'

# 画像とピークの推定位置を読み込む
image = load_dat_image(dat_path, image_width, image_height, cmap)
image = image.resize((scaled_width, scaled_height), Image.LANCZOS)  # 画像をスケーリング
peak_positions = pd.read_csv(csv_path)

# Tkinterのウィンドウを作成
root = tk.Tk()
root.title("Peak Position Adjuster")

# スクロールバーを設定
frame = Frame(root, width=scaled_width, height=scaled_height)
frame.pack(expand=True, fill=tk.BOTH)

canvas = Canvas(frame, width=scaled_width, height=scaled_height, scrollregion=(0, 0, scaled_width, scaled_height))
hbar = Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
hbar.pack(side=tk.BOTTOM, fill=tk.X)
vbar = Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
vbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
canvas.pack(expand=True, fill=tk.BOTH)

# 画像をキャンバスに描画
tk_image = ImageTk.PhotoImage(image)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

# ピーク位置を示すマーカーを描画
markers = []
texts = []
for _, row in peak_positions.iterrows():
    x, y = denormalize_coordinates(row['Posix'], row['Posiy'], scaled_width, scaled_height)
    color = 'yellow' if row['IDx'] % 5 == 0 or row['IDy'] % 5 == 0 else 'red'
    marker = canvas.create_oval(x-3, y-3, x+3, y+3, fill=color)
    text = canvas.create_text(x, y, text=f'{row["IDx"]},{row["IDy"]}', anchor=tk.NW, fill='white', font=("Helvetica", 7))
    markers.append((marker, text, row['IDx'], row['IDy']))
    texts.append(text)

# ドラッグ＆ドロップでマーカーの位置を調整する
def on_marker_drag(event, marker, text, id_x, id_y):
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    
    canvas.coords(marker, canvas_x-3, canvas_y-3, canvas_x+3, canvas_y+3)
    canvas.coords(text, canvas_x, canvas_y)
    for i, (m, t, idx, idy) in enumerate(markers):
        if m == marker:
            norm_x, norm_y = normalize_coordinates(canvas_x, canvas_y, scaled_width, scaled_height)
            peak_positions.at[i, 'Posix'] = norm_x
            peak_positions.at[i, 'Posiy'] = norm_y
            break

for marker, text, id_x, id_y in markers:
    canvas.tag_bind(marker, '<B1-Motion>', lambda event, m=marker, t=text, idx=id_x, idy=id_y: on_marker_drag(event, m, t, idx, idy))
    canvas.tag_bind(text, '<B1-Motion>', lambda event, m=marker, t=text, idx=id_x, idy=id_y: on_marker_drag(event, m, t, idx, idy))

# テキストの表示非表示を切り替える関数
text_visible = True
def toggle_text():
    global text_visible
    text_visible = not text_visible
    for text in texts:
        canvas.itemconfigure(text, state='normal' if text_visible else 'hidden')

# 画像の拡大縮小機能を実装する関数
current_scale = 1.0
def zoom(event, zoom_in=True):
    global current_scale, scaled_width, scaled_height, tk_image, image_on_canvas
    scale_factor = 1.1 if zoom_in else (1 / 1.1)
    new_scale = current_scale * scale_factor
    
    # 最大3段階までのズーム制限を設定
    if new_scale < 0.5 or new_scale > 3.0:
        return

    current_scale = new_scale
    scaled_width = int(image_width * current_scale)
    scaled_height = int(image_height * current_scale)

    # 画像をスケーリング
    resized_image = image.resize((scaled_width, scaled_height), Image.LANCZOS)
    tk_image = ImageTk.PhotoImage(resized_image)

    # 画像を再描画
    canvas.itemconfig(image_on_canvas, image=tk_image)
    canvas.config(scrollregion=(0, 0, scaled_width, scaled_height))

    # マーカーの位置を再計算して描画
    for marker, text, id_x, id_y in markers:
        x, y = denormalize_coordinates(peak_positions.loc[(peak_positions['IDx'] == id_x) & (peak_positions['IDy'] == id_y), 'Posix'].values[0],
                                       peak_positions.loc[(peak_positions['IDx'] == id_x) & (peak_positions['IDy'] == id_y), 'Posiy'].values[0],
                                       scaled_width, scaled_height)
        canvas.coords(marker, x-3, y-3, x+3, y+3)
        canvas.coords(text, x, y)

canvas.bind_all("<Control-Button-1>", lambda event: zoom(event, zoom_in=True))
canvas.bind_all("<Control-Button-3>", lambda event: zoom(event, zoom_in=False))

# カーソル座標を表示するラベルを作成
cursor_label = tk.Label(root, text="Cursor Position: (0.0, 0.0)")
cursor_label.pack()

# カーソルの位置を更新する関数
def update_cursor_position(event):
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    norm_x, norm_y = normalize_coordinates(canvas_x, canvas_y, scaled_width, scaled_height)
    cursor_label.config(text=f"Cursor Position: ({norm_x:.4f}, {norm_y:.4f})")

canvas.bind("<Motion>", update_cursor_position)

# 修正後のピーク位置を保存する関数
def save_positions():
    peak_positions.to_csv('adjusted_peaks.csv', index=False)
    print("Positions saved to 'adjusted_peaks.csv'")

# 保存ボタンを作成
save_button = tk.Button(root, text="Save", command=save_positions)
save_button.pack()

# テキスト表示非表示切り替えボタンを作成
toggle_text_button = tk.Button(root, text="Toggle Text", command=toggle_text)
toggle_text_button.pack()

# Tkinterのメインループを開始
root.mainloop()