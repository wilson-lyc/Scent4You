# Scent4You: AI-powered Perfume Recommendation System

Scent4You is a deep learning-based perfume recommendation system. By building Note Embedding (NEM) and Perfume Embedding (PEM) models, we map discrete fragrance ingredients and complex perfume structures into continuous vector spaces. This enables the system to understand concrete notes like "rose" and "sandalwood" and capture abstract olfactory semantics such as "woody" and "fresh," providing precise recommendations from natural language descriptions.

## Project Structure

| Folder/File | Description | Purpose |
| :--- | :--- | :--- |
| **Core** | | |
| Scent4You.ipynb | Main entry | Integrates data cleaning, visualization, and recommendation demos. |
| **Modules** | | |
| 1976_data_*.ipynb | Data processing | Crawling (collect) and cleaning (clean) scripts. |
| note_class_*.ipynb | Note mapping | Note family classification and mapping logic. |
| *_embedding_*.ipynb | Model training | Train Note Embedding and Perfume Embedding models. |
| *_recommendation_*.ipynb | Recommendation | Implement weighted average and neural network strategies. |
| vs.ipynb | Visualization | In-depth market and data analysis. |
| **Resources** | | |
| data/ | Datasets | Cleaned perfume data (.csv) and embedding dictionaries. |
| models/ | Model weights | Pretrained PyTorch models (.pth). |
| html/, img/ | Static assets | Generated interactive charts and images. |

## Technical Architecture
1. Note Embedding (NEM): Fine-tuned bert-base-chinese converts note names to 768-dimensional vectors; after convergence over 10 epochs, the epoch-9 model is used to generate embeddings.

![Weighted Average Recommendation Flow](img/加权平均推荐流程.png)

2. Perfume Embedding (PEM): A hierarchical network (top/middle/base sub-networks, attention aggregation, and multi-head attention) produces a 128-dimensional perfume embedding for classification and recommendation.

![PEM Recommendation Flow](img/PEM推荐流程.drawio.png)

3. Recommendation Strategy: Use cosine similarity to match vectorized perfumes with vectorized user requirements; implementations include a weighted average (top/middle/base weights 0.3/0.4/0.3) or a PEM-based neural network for higher accuracy.


## Data Pipeline

![Data Pipeline](img/模型数据管道.png)

Input fragrance text is converted to note embeddings via NEM. Top/middle/base notes are mapped to vectors and fed into PEM for feature extraction and attention aggregation. Multi-head attention fuses inter-layer interactions into a unified perfume embedding used for classification and similarity search.

## Quick Start
1. Prerequisites: Install core dependencies.
     ```bash
     pip install torch transformers scikit-learn pandas numpy matplotlib seaborn plotly networkx selenium webdriver-manager beautifulsoup4 tqdm wordcloud
     ```
2. Running the Project: Use Jupyter as the main interface.
     - Start Jupyter Lab:
         ```bash
         jupyter lab
         ```
     - Open Scent4You.ipynb (main entry) and execute cells in order.
     - Note: The Data Collection section contains web crawling code (time-consuming and network-dependent). By default, skip it and use preprocessed data in the data/ directory.
3. Usage Example: Find perfumes featuring "orange", "coffee", and "chocolate".
     ```python
     # 1. Load Model & Data (ensure prerequisite cells for the Recommendation Model section are executed)
     # 2. Define your preferences
     like_top_notes = ["橙子"]   # Orange
     like_mid_notes = ["咖啡"]   # Coffee
     like_base_notes = ["巧克力"] # Chocolate

     # 3. Get Top-10 recommendations
     rc_perfumes = get_recommended_perfumes(like_top_notes, like_mid_notes, like_base_notes, top_k=10)
     display(rc_perfumes)
     ```

