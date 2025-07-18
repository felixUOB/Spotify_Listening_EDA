# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 🎵 What Makes a Song Popular? An Exploratory Analysis of Spotify Audio Features | EDA

# ## 1.1. Introduction

# ## 1.2. Setup

# ### 1.2.1. Imports

from tqdm.notebook import tqdm
import pandas as pd
import plotnine as gg
import time
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# ### 1.2.2. Data Loading

# +
df = pd.read_csv("../data/dataset.csv")

df.shape
# -

# ## 1.3. Data Overview | Data Cleaning

# ### 1.3.1. Data Structure

df.head(5)

# ### 1.3.1. Data Cleaning

# +
df_clean = df.drop(columns=[
    'index',  
    'track_id',        
    'album_name',      
    'explicit',
    'artists',
])

df_clean.reset_index(drop=True, inplace=True)
# -

# ## 1.4. Analysis

#  ### 1.4.1. Popularity and Duration

# +
df_clean['duration_min'] = df_clean['duration_ms'] / 60000

fig, ax = plt.subplots(1, 2, figsize=(15.5, 4))
sns.histplot(data = df_clean['popularity'], bins = 50, ax=ax[0], color='orange')
ax[0].set_title('Popularity Distribution')

sns.histplot(df_clean['duration_min'], bins=50, ax=ax[1], color='green', binrange=(0,20))
ax[1].set_title('Track Duration Distribution')
plt.show()

print("Track Duration Mean : " + str(df_clean["duration_min"].mean()))
# -

# **Popularity Distribution**
# - The distribution shows a **heavy skew toward 0**, indicating a large number of tracks with small niches or poor-quality songs that few listeners enjoy.
# - Ignoring this initial spike, the remaining tracks follow a **slightly left-skewed curve**, suggesting that listeners may be more selective or reserved in awarding higher popularity scores, with fewer songs reaching the top end of the scale.

# **Track Duration Distribution**
# - The distribution shows a **predicatble normal curve** with a mean of **~3.8 mins** aligns with typical duration of mainstream songs.
# - The distribution shows a strong peak around **3–4 minutes**, with a small number of outliers that are either very short (<1 min, possibly interludes) or very long (>10 min, likely live or experimental tracks).

sns.scatterplot(data=df_clean, x='duration_min', y='popularity', alpha=0.5)

# **Relationship**
#
# - There appears to be **no strong correlation** between a **track’s duration** and its **popularity** for the majority of songs.
#
# - However, when looking at outliers in track length (very long songs), there seems to be a **pseudo-ceiling around ~60 popularity**, suggesting that listeners generally show less enthusiasm for extremely long tracks, even if they’re well-regarded.

# ### 1.4.2. Audio Feature Exploration

# +
popular_threshold = 60

df_clean['is_popular'] = df_clean['popularity'] > popular_threshold

feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']

feature_means = df_clean.groupby('is_popular')[feature_cols].mean().T
print(feature_means)
# -

feature_means.plot(kind='bar', figsize=(10,5), color=['gray','orange'])
plt.title("Average Audio Features: Popular vs Non-Popular Songs")
plt.ylabel("Mean Feature Value")
plt.show()

# **Feature Analysis** 
# **Danceability, energy, and valence** all show higher average values in popular songs, suggesting that tracks that are more rhythmically engaging, energetic, and emotionally positive are better received. This aligns with mainstream trends where upbeat, feel-good tracks dominate charts and playlists.
#
# On the other hand, **acousticness, instrumentalness, liveness, and speechiness** are more common in less popular songs, which may reflect:
# - Acoustic & instrumental tracks often belong to niche genres with smaller audiences. Tracks with high liveness (live recordings) and speechiness (spoken word/rap-heavy) may have limited mainstream exposure.
# - Radio and playlist algorithms tend to prioritize polished, energetic studio recordings over experimental or live material.

# +
corr = df_clean[feature_cols + ['popularity']].corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Between Audio Features & Popularity")
plt.show()
# -

# **Correlation Analysis**
#
# - There appears to be a mild positive correlation between danceability, energy, and popularity, suggesting that tracks with more rhythmic appeal and higher intensity are slightly more likely to gain mainstream attention.
# - This likely reflects mainstream listener preferences for accessible, upbeat music rather than niche or experimental works.
#
# - Conversely, there is a moderate negative correlation between instrumentalness and popularity, indicating that purely instrumental tracks tend to attract less mainstream appeal, possibly due to their limited exposure on popular playlists and radio.
# - Other features such as acousticness, liveness, and speechiness also show weak or negligible correlations, suggesting they play a smaller role in determining a track’s popularity.


