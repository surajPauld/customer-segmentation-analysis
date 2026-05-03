# customer-segmentation-analysis
Applied unsupervised learning to segment customers using the Mall Customers dataset. Implemented clustering (K-Means, Hierarchical) on age, income, and spending score with preprocessing and scaling. Derived actionable insights for targeted marketing and customer retention strategies.

1. Problem Statement:

Businesses want to understand different types of customers without predefined labels.
Use unsupervised learning to segment customers based on their behavior.
system can also be deployed as a web-based healthcare application for real-world use.


2. Dataset Used:

I have used “Mall Customers Segmentation” data set from Kaggle
Link: https://www.kaggle.com/datasets/abdallahwagih/mall-customerssegmentation?resource=download


3.Algorithms used:

● K-Means
● Hierarchical Clustering
● DBSCAN (optional – bonus)


All the tasks performed:
Data Processing, K-Means and determining Optimal k = 4.


Hierarchical Clusturing:

Visualizing Clusters:

For Buisness insight report:

As we can see:

• Cluster 0:
Old, mid income, bellow avg spending – “conservative old shoppers”.
• Cluster 1:
Middle-aged, high income, high spending – “premium shoppers”.
• Cluster 2:
Young, mid income, mid spending – “conservative young shoppers”.
• Cluster 3:
Middle-aged, high income, low spending – “conservative middle aged
shoppers”.


Problems faced:

• Selecting the right input features was challenging because different combinations (e.g.,
Age–Income–Spending vs only Income–Spending) produced different cluster shapes and
business interpretations, even though the task was the same.

• Scaling was mandatory and required careful checking, because K-Means and
Hierarchical clustering are distance-based and unscaled features can dominate the
distance computation and distort clusters.

• Choosing the “optimal” number of clusters 𝑘 was not straightforward: the Elbow curve
can be gradual (no clear bend) and the Silhouette score may favor a different 𝑘, so we
had to balance statistical scores with interpretability.

• Hierarchical clustering required deciding the linkage method and where to cut the
dendrogram; small changes in cut level changed the final grouping, which made
comparison with K-Means slightly inconsistent.

• Cluster visualization in 2D/3D sometimes hid structure because projecting multiple
features into 2D can overlap points, making cluster boundaries look less clear than they
are in higher-dimensional space.

• Translating clusters into business terms was a key difficulty because clustering gives
groups without labels, so we had to infer meaningful segments by comparing cluster
averages (income/spending/age) and ensuring the segments were actionable.


Conclusion:

This project successfully applied unsupervised learning to segment customers without predefined
labels by using clustering methods (K-Means and Hierarchical clustering) after mandatory feature
scaling.

The optimal number of clusters was selected using cluster evaluation techniques (Elbow method
and Silhouette score), and the resulting clusters were visualized in 2D/3D to understand
separation and overlap.

By profiling each cluster using average customer attributes (such as age, annual income, and
spending score), we interpreted the groups as practical business segments that can support
targeted marketing, personalized offers, and customer retention strategies.

Overall, the work demonstrated the difference between supervised and unsupervised learning and
reinforced how distance metrics and evaluation scores influence clustering quality and
interpretability.
