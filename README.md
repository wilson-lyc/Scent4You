# Scent4You: AI 驱动的香水推荐系统

**Scent4You** 是一个基于深度学习的香水推荐系统。通过构建 **Note Embedding (NEM)** 和 **Perfume Embedding (PEM)** 模型，我们将离散的香料成分和复杂的香水结构映射为连续的向量空间表示。这使得系统不仅能理解“玫瑰”、“檀香”等具体香料成分，还能捕捉“木质”、“清新”等抽象的嗅觉语义，从而根据用户的自然语言描述进行精准推荐。

## 项目结构
项目结构如下：
| 目录/文件 | 描述 | 用途 |
| :--- | :--- | :--- |
| **Core** | | |
| `Scent4You.ipynb` | **主入口** | 整合数据清洗、可视化分析与推荐模型演示的完整流程。 |
| **Modules** | | |
| `1976_data_*.ipynb` | 数据处理 | 包含爬虫 (`collect`) 和清洗 (`clean`) 脚本。 |
| `note_class_*.ipynb` | 香调映射 | 处理香调分类与映射逻辑。 |
| `*_embedding_*.ipynb` | 模型训练 | 训练 Note Embedding 和 Perfume Embedding 模型。 |
| `*_recommendation_*.ipynb` | 推荐算法 | 实现加权平均和神经网络两种推荐策略。 |
| `vs.ipynb` | 可视化 | 市场与数据的深度可视化分析。 |
| **Resources** | | |
| `data/` | 数据集 | 存放清洗后的香水数据 (`.csv`) 与 Embedding 字典。 |
| `models/` | 模型权重 | 存放预训练好的 PyTorch 模型文件 (`.pth`)。 |
| `html/`, `img/` | 静态资源 | 存放生成的交互式图表和图片。 |

## 技术架构
### 1. Note Embedding (NEM)
基于微调的 **bert-base-chinese** 将香料名称转换为 768 维向量，并在 10 个 epoch 收敛后选用第 9 epoch 模型生成向量。

![加权平均推荐流程](img/加权平均推荐流程.png)

### 2. Perfume Embedding (PEM)
通过层级神经网络（前/中/后调子网络、注意力聚合与多头注意力交互）生成代表整瓶香水的 128 维 `perfume embedding`，并用于香调分类与推荐。

![PEM 推荐流程](img/PEM推荐流程.drawio.png)

### 3. 推荐策略
使用余弦相似度比较向量方向，将香水与用户需求向量化后进行匹配；实现上可采用加权平均（前/中/后调权重 0.3/0.4/0.3）或基于 PEM 的神经网络以获得更准确的结果。


## 数据传输流程

![数据管道](img/模型数据管道.png)

概述：输入的香料文本经 NEM 生成 note embedding 字典，前/中/后调映射为向量后进入 PEM 做特征提取与注意力聚合，并经多头注意力完成层间交互融合为统一的 `perfume embedding`，用于分类与相似度检索。

## 快速开始

### 1. 环境准备 (Prerequisites)
为能够正常运行本项目。请确保安装以下核心依赖库：

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn plotly networkx selenium webdriver-manager beautifulsoup4 tqdm wordcloud
```
### 2. 运行指南
本项目采用 Jupyter Notebook 作为主要的交互界面。

1.  启动 Jupyter Lab 或 Jupyter Notebook：
    ```bash
    jupyter lab
    ```
2.  打开 **`Scent4You.ipynb`**。这是项目的主入口文件。
3.  按照 **`Scent4You.ipynb`** 中的章节顺序执行代码。
    - **注意**：**`Data Collection`** 章节包含爬虫代码，耗时较长且依赖网络环境；默认建议跳过，直接使用 `data/` 目录下预处理好的数据。

### 3. 使用示例

以下代码展示了如何在 Notebook 中寻找一款带有“橙子”、“咖啡”和“巧克力”元素的香水：

```python
# 1. 加载模型与数据 (Load Model & Data)
# (确保已运行 Recommendation Model 章节的前置单元格)

# 2. 定义你的偏好 (Define your preferences)
like_top_notes = ["橙子"]   # Orange
like_mid_notes = ["咖啡"]   # Coffee
like_base_notes = ["巧克力"] # Chocolate

# 3. 获取 Top 10 推荐 (Get recommendations)
rc_perfumes = get_recommended_perfumes(like_top_notes, like_mid_notes, like_base_notes, top_k=10)
display(rc_perfumes)
```

