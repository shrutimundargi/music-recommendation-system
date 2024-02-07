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

### Data Preprocessing and Feature Selection

![111sds](https://github.com/shrutimundargi/music-recommendation-system/assets/48567754/98af4564-4e18-407b-b4cc-430727a0d27d)

Data cleaning plays a pivotal role in the preparation of data for subsequent analysis. PySpark offers a range of functions, including regexp_replace and trim, which prove invaluable in the removal of unwanted characters and overall data cleansing process. These functions efficiently eliminate inconsistencies and ensure the integrity of the dataset. Furthermore, pre-processing operations such as splitting arrays and expanding columns are essential steps undertaken to ready the data for analysis. By effectively manipulating the data structure, these operations streamline the dataset, making it more amenable to analytical processes and enhancing the accuracy of subsequent analyses.

An enhancement was made to the feature matrix with the aim of enhancing the performance of the music recommendation system. Specifically, the 'release_date' column was leveraged to extract the year information, thereby creating a new 'year' feature. This transformation was achieved utilizing the PySpark function 'year', which effectively isolated the year component from the release dates.

Subsequently, the resulting DataFrame, now enriched with the newly engineered 'year' feature, served as the basis for constructing the feature matrix. Notably, prior to this step, the 'id' and 'name' columns were dropped from the DataFrame, ensuring that only relevant features were included in the final matrix. This meticulous process ensures the integrity and effectiveness of the music recommendation system's feature matrix.


### K-means Clustering: 

K-means stands as a prominent clustering algorithm renowned for its ability to partition datasets into a predetermined number of clusters, denoted as 'k'. This segmentation is achieved by assessing the distance between data points and the centroids of the clusters. The algorithm operates iteratively, initially assigning data points to the nearest centroid and subsequently updating the centroids based on the mean of the data points assigned to each cluster.

The iterative process continues until the centroids converge to a stable solution, indicating that further adjustments yield minimal change in cluster assignments. This convergence marks the completion of the algorithm's execution.

Widely employed across diverse domains encompassing machine learning, image processing, and marketing segmentation, K-means offers notable utility, particularly when the number of clusters is either predetermined or can be estimated via heuristic techniques. Despite its widespread application, K-means exhibits sensitivity to the initial selection of centroids, a characteristic that can occasionally lead to convergence towards local optima rather than the global optimum solution. This sensitivity underscores the importance of carefully initializing the algorithm to mitigate the risk of suboptimal clustering outcomes.

<img width="431" alt="Screenshot 2024-02-07 at 12 12 29 PM" src="https://github.com/shrutimundargi/music-recommendation-system/assets/48567754/9da4dc95-4a4c-4c30-a85a-f0de0fc1d2a0">

#### Optimizing K-means:
To determine the optimal number of clusters within a dataset through K-means clustering. Initially, the data undergoes scaling utilizing the mean and standard deviation of each feature to ensure uniformity and facilitate accurate clustering.

Subsequently, the function iterates over a specified range of clusters, typically spanning from 10 to 30, and computes the Within-Cluster-Sum-of-Squares (WCSS) for each cluster configuration employing the K-means algorithm. WCSS, representing the sum of squared distances between each data point and its assigned centroid within a cluster, serves as a metric to evaluate clustering performance.

Upon computing the WCSS values for each cluster count, the function proceeds to visualize these values plotted against the corresponding number of clusters. This graphical representation enables the application of the elbow method, a heuristic approach utilized to identify the optimal number of clusters. By inspecting the plot, analysts can discern the point where the rate of decrease in WCSS begins to plateau, indicative of diminishing returns in clustering efficacy. This inflection point, often resembling an elbow shape in the plot, serves as a guideline for selecting the optimal number of clusters for subsequent analysis.

<img width="380" alt="Screenshot 2024-02-07 at 12 37 39 PM" src="https://github.com/shrutimundargi/music-recommendation-system/assets/48567754/957e8972-5105-4ddf-94bb-5e8f1d8dac35">

### Fuzzy C-Means Clustering:

Fuzzy C-Means (FCM) represents a sophisticated clustering algorithm renowned for its unique approach to assigning data points to clusters based on similarity measures. Unlike conventional clustering methods, FCM introduces the concept of fuzzy membership, allowing data points to belong to multiple clusters simultaneously rather than being strictly assigned to a single cluster.

The algorithm operates iteratively, continuously updating both the centroids of the clusters and the degree of membership of each data point in every cluster. This iterative refinement process continues until the membership values converge to a stable solution, indicating that further adjustments yield minimal changes in cluster assignments.

FCM finds extensive applications across diverse domains, including image segmentation, pattern recognition, and data mining. Its distinctive capability to accommodate fuzzy memberships proves particularly advantageous when dealing with datasets characterized by overlapping or ambiguous boundaries, where traditional clustering approaches may struggle to provide accurate segmentation or classification. Thus, FCM stands as a valuable tool for extracting meaningful insights from complex datasets, contributing significantly to various analytical endeavors.

#### Optimizing Fuzzy C-Means:

The Fuzzy Partition Coefficient (FPC) is a metric commonly used in Fuzzy C-Means (FCM) clustering to assess the quality of clustering solutions. It measures the degree of fuzziness in the clustering, indicating how well the data points are distributed among the clusters. A higher FPC value suggests that the clustering is more robust and distinct. By leveraging the FPC and the elbow method, you can effectively determine the appropriate number of clusters for FCM clustering, ensuring robust and meaningful partitioning of your dataset.

<img width="378" alt="Screenshot 2024-02-07 at 12 40 42 PM" src="https://github.com/shrutimundargi/music-recommendation-system/assets/48567754/a9d877b9-ecd4-4189-9a4e-4d43fca88e84">


### Gaussian Mixture Model

The Gaussian Mixture Model (GMM) stands as a probabilistic model fundamental for understanding datasets assumed to originate from a mixture of Gaussian distributions. Its primary objective lies in estimating the parameters characterizing these underlying Gaussian distributions, encompassing means, covariance matrices, and mixture weights dictating the relative contributions of each distribution.

GMM's versatility extends to a spectrum of applications, including clustering, density estimation, and feature extraction, rendering it a cornerstone in various analytical tasks. Notably, the algorithm's ability to discern intricate patterns within datasets makes it particularly adept for tasks involving complex data structures.

Training a GMM typically involves leveraging the Expectation-Maximization (EM) algorithm. This iterative procedure iteratively updates the model's parameters to maximize the likelihood of observing the dataset given the model assumptions. Through alternating between expectation and maximization steps, EM gradually refines the parameter estimates until convergence, effectively capturing the underlying structure of the data.

In summary, GMM serves as a powerful tool in statistical modeling and inference, offering a robust framework for uncovering latent structures within datasets and facilitating a deeper understanding of their inherent complexities.

#### Optimizing Gaussian Mixture Model:

In the process of estimating the optimal number of clusters using the Bayesian Information Criterion (BIC), the following steps are executed:

Feature Scaling: The numerical features are standardized using the StandardScaler method to ensure uniformity in their distributions, enhancing the effectiveness of the subsequent clustering analysis.

Candidate Cluster Counts Generation: A range of potential numbers of clusters is generated, providing a spectrum of choices for the clustering algorithm to evaluate. This range encompasses a variety of cluster counts, enabling comprehensive exploration of clustering solutions.

GMM Model Fitting and BIC Calculation: For each candidate number of clusters, a Gaussian Mixture Model (GMM) is fitted to the data. Subsequently, the BIC score for each model is computed utilizing the gmm.bic() method. The BIC serves as a metric for evaluating the quality of fit of each model, balancing the trade-off between model complexity and goodness of fit.

Optimal Cluster Selection: The BIC scores obtained for the different numbers of clusters are plotted on a line plot. The plot visually depicts the relationship between the number of clusters and the corresponding BIC scores. By analyzing the plot, the optimal number of clusters is determined based on the inflection point in the curve. This inflection point signifies a favorable compromise between model complexity and the fidelity of the clustering solution, thus representing an optimal choice for the number of clusters.

Overall, this approach leverages the BIC criterion to guide the selection of the optimal number of clusters, ensuring that the resulting clustering solution strikes an appropriate balance between model complexity and the quality of fit to the data.

<img width="427" alt="Screenshot 2024-02-07 at 12 44 19 PM" src="https://github.com/shrutimundargi/music-recommendation-system/assets/48567754/20192c77-9e3a-4d9c-a2d7-3e6ed48d2d3b">

## Song Extraction

Spotify stands as a widely acclaimed music streaming platform boasting an extensive library comprising millions of songs spanning various genres and artists. Central to the accessibility and programmability of Spotify's vast musical ecosystem is the Spotify API.

The Spotify API serves as a comprehensive toolkit empowering developers to interact with Spotify's rich music data and functionalities programmatically. Through this interface, developers can seamlessly access a plethora of song information, including titles, artists, albums, and durations, directly from Spotify's metadata repository.

This API facilitates seamless integration of Spotify's music data and functionalities into diverse applications, websites, and services developed by third-party creators. By leveraging the Spotify API, developers can augment their platforms with Spotify's extensive music catalog and features, enriching the user experience with personalized, high-quality audio content.

![sdsf](https://github.com/shrutimundargi/music-recommendation-system/assets/48567754/85109889-5d52-4cee-9a66-b5f990fcaad1)

## Recommendation

In this particular use case, we focus on a specific song, "All of Me," and aim to receive recommendations for additional songs based on this input.

The process involves leveraging recommendation systems, which utilize algorithms to analyze user preferences and behavior to generate personalized song suggestions. By inputting "All of Me" into the recommendation system, the algorithm identifies songs that share similar characteristics or appeal to users who enjoyed "All of Me."

Upon submitting the song, the recommendation system processes the input and generates a list of recommended songs tailored to the user's preferences and the attributes of "All of Me." These recommendations may encompass tracks from the same artist, songs within a similar genre, or compositions with comparable musical features such as tempo, mood, or instrumentation.

Ultimately, the user receives a curated list of song recommendations, providing an enriched listening experience and potentially introducing them to new music that aligns with their tastes and preferences.

![gdgfdx](https://github.com/shrutimundargi/music-recommendation-system/assets/48567754/3f0f3d17-69ad-459a-a4f5-bd4eba65f141)


### Results

<img width="763" alt="Screenshot 2024-02-07 at 12 46 25 PM" src="https://github.com/shrutimundargi/music-recommendation-system/assets/48567754/1d186d57-bc4c-4b1f-9efd-b5d66e67d9f3">

## Conclustion

In our project, we endeavored to construct a music recommendation system employing various clustering algorithms, namely K Means, Gaussian Mixture Model (GMM), and Fuzzy C Means. Our objective was to assess the performance of these algorithms and determine the most suitable approach for our dataset.

Upon thorough evaluation, we found that K Means exhibited the least computation time compared to GMM and Fuzzy C Means, rendering it the most efficient algorithm for our specific dataset. This efficiency is crucial for ensuring real-time or near-real-time recommendations in practical applications.

Furthermore, after comparing the quality of recommendations produced by each algorithm, we concluded that K Means delivered satisfactory results, aligning closely with our project objectives and user expectations. While GMM and Fuzzy C Means may offer alternative benefits in certain contexts, such as handling complex data distributions or accommodating fuzzy memberships, the simplicity and efficiency of K Means proved to be the most effective solution for our music recommendation system.

In summary, based on our comprehensive analysis and performance evaluation, we determine that K Means stands as the optimal algorithm for our music recommendation system, offering a balance of computational efficiency and satisfactory recommendation quality tailored to our dataset and project requirements.

<img width="415" alt="Screenshot 2024-02-07 at 12 47 27 PM" src="https://github.com/shrutimundargi/music-recommendation-system/assets/48567754/87af65bf-87c5-4756-94e0-02ed3e5e98c0">
