# music-recommendation-system

## Introduction

### Background:

It might be overwhelming to decide what to play next in the modern world where streaming music has become so simple and there are millions and millions of songs available to listen to. Users may find it difficult and time-consuming to find new tracks that suit their tastes. It is also probable that the user would wind up merely choosing what to play next rather than really listening to any songs when there are millions of alternatives available in a genre, which could be frustrating. Systems that generate recommendations make this work incredibly simple.

### Motivation:

Machine learning algorithms can be used to analyze and propose music based on a variety of characteristics, including user listening habits and tastes. Music recommendation systems can offer personalized suggestions that are appropriate for the user's musical preferences and can simplify things for users by giving them an easy way to find new artists and genres they might enjoy. It can help consumers discover musical genres they might like and improve their overall listening experience.

## Dataset Description

This project utilized the Spotify dataset (202MB), which was available on Kaggle. The dataset contained metadata for approximately 600,000 songs, comprising a total of 19 columns with parameters such as tempo, valence, artists, year, etc. Dataset download link: https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks

## Methodology

By calculating the distance between songs that a user had previously listened to and new songs using a variety of variables in our dataset, we intended to develop a content-based music recommendation system. We aimed to recommend to the user a small number of songs that had the smallest distance, effectively locating songs in the same genre as the user's past preferences by calculating the distance. We believed that clustering algorithms (examples of which were provided below) would be able to provide us with respectable results.

### Data Preprocessing

Data cleaning plays a pivotal role in the preparation of data for subsequent analysis. PySpark offers a range of functions, including regexp_replace and trim, which prove invaluable in the removal of unwanted characters and overall data cleansing process. These functions efficiently eliminate inconsistencies and ensure the integrity of the dataset. Furthermore, pre-processing operations such as splitting arrays and expanding columns are essential steps undertaken to ready the data for analysis. By effectively manipulating the data structure, these operations streamline the dataset, making it more amenable to analytical processes and enhancing the accuracy of subsequent analyses.



### K-means Clustering: 

The K-means technique presupposes k centroids, where a centroid is a hypothetical or actual place denoting the cluster's center. After that, it assigns each data point to the closest cluster while attempting to minimize the size of the centroids. Each data point is assigned to a certain cluster by minimizing the total of squares within each cluster.

DBSCAN Clustering: The foundational algorithm for density-based clustering is called Density-Based Spatial Clustering of Applications with Noise (DBSCAN). With a big amount of data that contains noise and outliers, it can find clusters of various sizes and forms.

Spectral Clustering: In graph theory, a method called spectral clustering is used to locate groups of nodes in a graph based on the edges that connect them. The approach is adaptable and enables the clustering of non-graph data as well. The information used in spectral clustering comes from the eigenvalues (spectrum) of unique matrices created from the graph or data collection.
