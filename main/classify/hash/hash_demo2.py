
import numpy as np

# [127, 150, 160]  平均向量（三维空间向量）
# [110, 80, 96]   L2 = 96.35870484808314
# [192, 160, 180]

# [127, 150, 105]
# [127, 50, 160]
# [30, 150, 160]

# [127, 150, 160], [192, 160, 180] 68.7386354243376
# [127, 150, 160], [127, 150, 105] 55.0
# [127, 150, 160], [127, 50, 160] 100.0
# [127, 150, 160], [30, 150, 160] 97.0

vector1 = np.array([127, 150, 160])
vector2 = np.array([192, 160, 180])
vector3 = np.array([127, 150, 105])
vector4 = np.array([127, 50, 160])
vector5 = np.array([30, 150, 160])

# op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
op2 = np.linalg.norm(vector1 - vector2)
op3 = np.linalg.norm(vector1 - vector3)
op4 = np.linalg.norm(vector1 - vector4)
op5 = np.linalg.norm(vector1 - vector5)
print("[127, 150, 160], [192, 160, 180]", op2)
print("[127, 150, 160], [127, 150, 105]", op3)
print("[127, 150, 160], [127, 50, 160]", op4)
print("[127, 150, 160], [30, 150, 160]", op5)
# print(op2)
