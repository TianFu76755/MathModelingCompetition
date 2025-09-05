import pandas as pd
import matplotlib.pyplot as plt
from Data.DataManager import DM

df1 = DM.get_data(1)
df2 = DM.get_data(2)

# 可视化两个表格的曲线
plt.figure(figsize=(10,6))
plt.plot(df1["波数 (cm-1)"], df1["反射率 (%)"], label="Incidence 10° (df1)", alpha=0.8)
plt.plot(df2["波数 (cm-1)"], df2["反射率 (%)"], label="Incidence 15° (df2)", alpha=0.8)

# plt.gca().invert_xaxis()  # 光谱学中常常是波数从大到小
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Reflectance (%)")
plt.title("Infrared Interference Spectra of 4H-SiC Epitaxial Layer")
plt.legend()
plt.grid(True)
plt.show()
