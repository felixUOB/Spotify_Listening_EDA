# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🎵 Spotify Tracks Attributes and Popularity | EDA

# %% [markdown]
# ## 1.1. Introduction

# %% [markdown]
# ## 1.2. Setup

# %% [markdown]
# ### 1.2.1. Imports

# %%
from tqdm.notebook import tqdm
import pandas as pd
import plotnine as gg
import time
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% [markdown]
# ### 1.2.2. Data Loading
# %%
df = pd.read_csv("../data/dataset.csv")

df.shape

# %% [markdown]
# ## 1.3. Data Overview | Data Cleaning

# %% [markdown]
# ### 1.3.1. Data Structure

# %%
df.head(5)

df.dtypes

# Null value analysis - how dealt with

# %%

# %% [markdown]
# ## 1.4. Analysis

# %% [markdown]
# ### 1.4.1. Popularity and Duration

# %%
# Popularity Dist - Top tracks

# Duration Dist - Most common length

# %% [markdown]
# ### 1.4.2. Audio Feature Exploration

# %%
# Histograms - Descriptive Stats

# %% [markdown]
# ### 1.4.3. Key, Mode & Musicality

# %%
# Most common musical keys
# Proportion of major vs minor mode
# Does mode (major/minor) affect valence (happiness) or popularity?

# %% [markdown]
# ### 1.4.4. Tempo & Rhythm

# %%
# Tempo distribution → are most songs clustered around ~120 BPM?
# Relation between tempo and danceability or energy

# %% [markdown]
# ### 1.4.5. Other

# %%
# Genre Radar Charts – compare average audio profiles of genres
# Cluster tracks based on features (K-Means or PCA for dimensionality reduction)
# Predict popularity from audio features (simple regression model)

# %% [markdown]
# ## 1.5. Conclusion

# %%
