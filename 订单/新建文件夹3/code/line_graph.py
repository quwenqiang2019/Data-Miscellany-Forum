import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_dir)

df = pd.DataFrame(pd.read_excel(os.path.join(base_dir, 'data', '模型指标.xlsx')))
df = df.dropna()
print(df)
print(df.columns)

sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.plot(df['迭代次数=200+'], df['ESRGAN修改前PSNR指标'], label='ESRGAN修改前')
plt.plot(df['迭代次数=200+'], df['ESRGAN修改后PSNR指标'], label='ESRGAN修改后')
plt.plot(df['迭代次数=200+'], df['MSRGAN的PSNR指标'], label='MSRGAN')
plt.plot(df['迭代次数=200+'], df['原版RRDBNet的PSNR指标'], label='原版RRDBNet')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'PSNR.jpg'), bbox_inches='tight', dpi = 600)
plt.show()


plt.plot(df['迭代次数=200+'], df['ESRGAN修改前SSIM指标'], label='ESRGAN修改前')
plt.plot(df['迭代次数=200+'], df['ESRGAN修改后SSIM指标'], label='ESRGAN修改后')
plt.plot(df['迭代次数=200+'], df['MSRGAN的SSIM指标'], label='MSRGAN')
plt.plot(df['迭代次数=200+'], df['原版RRDBNet的SSIM指标'], label='原版RRDBNet')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'SSIM.jpg'), bbox_inches='tight', dpi = 600)
plt.show()