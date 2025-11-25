import numpy as np

EMB_PATH = r"D:\Learning\slfm\data\amazon\out\hyperbolic_embeddings.npy"

z = np.load(EMB_PATH)
print("shape:", z.shape)

finite_mask = np.isfinite(z)
print("all finite:", finite_mask.all())

# 统计 NaN / Inf 个数
num_nan = np.isnan(z).sum()
num_posinf = np.isposinf(z).sum()
num_neginf = np.isneginf(z).sum()
print("NaN:", num_nan, " +Inf:", num_posinf, " -Inf:", num_neginf)

# 看一下数值范围
abs_z = np.abs(z[np.isfinite(z)])
print("min|z|:", abs_z.min(), "max|z|:", abs_z.max())
