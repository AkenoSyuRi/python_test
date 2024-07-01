import numpy as np

np.set_printoptions(suppress=True)

"""
mic阵列远场条件:
阵列间距固定前提下，波长λ越大，f越小，需要的r就越小
- r >= 2*(D**2)/λ
- r >= 2*(D**2)*f/c

r: 信源与mic阵列之间的距离
D: mic阵列孔径(阵列2个mic之间最大间距)
λ: 信源波长
f: 信源频率
c: 声速
"""

c = 340
d = np.array([0.0175, 0.035, 0.07, 0.14, 0.28, 0.56]).reshape(-1, 1)
# f = np.array([10, 2000, 6000, 8000, 16000, 24000]).reshape(1, -1)
f = np.array([8000, 4900, 2450, 1225, 613, 300]).reshape(1, -1)

D = d * 2  # 圆阵最大间距为直径
r_min = 2 * np.square(D) * f / c

print("spacing: ", d.reshape(-1))
print("frequency: ", f.reshape(-1))
print("shape: (spacing, frequency), min distance for far field:")
# print(r_min)
print(np.diag(r_min))
...
