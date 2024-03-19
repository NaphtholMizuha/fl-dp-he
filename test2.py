import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title(r'$\sin(x)$')  # 使用 LaTeX 语法在标题中输入数学符号
plt.xlabel(r'$x$')  # 使用 LaTeX 语法在 x 轴标签中输入数学符号
plt.ylabel(r'$y$')  # 使用 LaTeX 语法在 y 轴标签中输入数学符号
plt.savefig('sin.png')