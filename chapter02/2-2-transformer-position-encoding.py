# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "japanize-matplotlib",
# ]
# ///

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np

K = 50  # 単語列の最大長
D = 64  # 埋め込みの次元

# 位置符号行列を初期化
pos_enc = np.empty((K, D))

for i in range(K):  # 単語位置iでループ
    for k in range(D // 2):  # kの値でループ
        theta = i / (10000 ** (2 * k / D))
        pos_enc[i, 2 * k] = np.sin(theta)
        pos_enc[i, 2 * k + 1] = np.cos(theta)

# 行列を画像で表示
im = plt.imshow(pos_enc)
plt.xlabel("次元")  # X軸のラベルを設定
plt.ylabel("位置")  # Y軸のラベルを設定
plt.colorbar(im)  # 値と色の対応を示すバーを付加
plt.show()

# 位置符号同士の内積を計算
dot_matrix = np.matmul(pos_enc, pos_enc.T)
# 行列を画像で表示
im = plt.imshow(dot_matrix, origin="lower")
plt.xlabel("位置")
plt.ylabel("位置")
plt.colorbar(im)
plt.show()
