<!-- 头部：动态打字机效果 -->
<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=30&pause=1000&color=F7931A&center=true&vCenter=true&width=800&lines=Dive+Into+Deep+Learning;From+Zero+to+Hero+in+AI;Code+%2B+Math+%2B+Theory;PyTorch+Implementation" alt="Typing SVG" />
</div>

<!-- 徽章区域 -->
<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
  <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" />
</div>

<br/>

<div align="center">
  <p align="center">
    <b>从感知机到大模型，动手学深度学习的完整旅程。</b>
    <br />
    <i>"Talk is cheap. Show me the code."</i>
  </p>
</div>

---

## 📖 前言：为什么是现在？

几年前，深度学习还只是大公司实验室里的秘密武器。那时，我们的父母不知道什么是机器学习，神经网络被认为是过时的工具。

**但在过去的五年里，世界被重塑了：**
*   🚗 **自动驾驶** 不再是科幻小说。
*   📧 **智能回复** 解放了我们的收件箱。
*   🧬 **AI for Science** 正在破解蛋白质折叠的秘密。
*   🧠 **AlphaGo** 在围棋上超越了人类最强棋手。

本书（及本项目）致力于填补一个巨大的空白：**将数学原理、代码实现和工程实践结合起来**。我们不希望你只看到公式，也不希望你只会调用 API。我们希望你**理解**，并能**从零实现**。

---

## ⏳ 深度学习简史：巨人的肩膀

深度学习并非一蹴而就。本项目将带你回顾这些里程碑时刻，并亲手实现它们背后的算法。

<div align="center">
<table align="center" style="border:none">
  <tr>
    <td align="center" width="33%">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Mark_I_Perceptron%2C_Figure_2_of_operator%27s_manual.png/640px-Mark_I_Perceptron%2C_Figure_2_of_operator%27s_manual.png" width="100%" alt="Perceptron"/>
      <br/>
      <b>1958: 感知机 (Perceptron)</b><br/>
      <sub>深度学习的黎明，Frank Rosenblatt 的 Mark I 机器。</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/AlexNet_architecture.png/800px-AlexNet_architecture.png" width="100%" alt="AlexNet"/>
      <br/>
      <b>2012: AlexNet 时刻</b><br/>
      <sub>ImageNet 竞赛的爆发，CNN 统治计算机视觉的开始。</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Lee_Sedol_%28B%29_vs_AlphaGo_%28W%29_-_Game_5.svg/426px-Lee_Sedol_%28B%29_vs_AlphaGo_%28W%29_-_Game_5.svg.png" width="100%" alt="AlphaGo"/>
      <br/>
      <b>2016: AlphaGo</b><br/>
      <sub>强化学习的巅峰，AI 战胜人类围棋冠军李世石。</sub>
    </td>
  </tr>
</table>
</div>

---

## 🗺️ 学习路线图

本项目分为三个核心部分，带你从入门到精通：

### 🟢 第一部分：基础与预备
> *万丈高楼平地起。*
*   **数学基础**：线性代数、微积分、概率论（只需基础 Python 知识）。
*   **自动微分**：深度学习框架的核心魔法。
*   **线性回归 & Softmax**：从零实现最简单的模型。

### 🔵 第二部分：现代深度学习技术
> *这是工业界最常用的工具箱。*
*   **CNN (卷积神经网络)**：ResNet, VGG, AlexNet —— 计算机视觉的基石。
*   **RNN (循环神经网络)**：处理时间序列和文本数据。
*   **Attention (注意力机制)**：Transformer 的前身，自然语言处理的革命。

### 🔴 第三部分：前沿与应用
> *走向大规模与落地。*
*   **优化算法**：Adam, SGD, 学习率调度。
*   **计算机视觉应用**：目标检测 (YOLO), 语义分割。
*   **NLP 与 预训练模型**：BERT, GPT 的原理与微调。

---

## 🛠️ 快速开始

本项目代码基于 **PyTorch**。我们封装了一个轻量级的 `d2l` 包，方便复用代码。

### 环境依赖
```python
import torch
from torch import nn
from d2l import torch as d2l
