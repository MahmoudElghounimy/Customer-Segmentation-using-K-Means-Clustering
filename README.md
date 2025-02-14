### **Customer Segmentation Using K-Means Clustering**  



---

### **1. Load and Explore the Data**  
- Reads the `Mall_Customers.csv` dataset.  
- Checks the dataset structure, shape, and missing values.  

### **2. Select Features for Clustering**  
- Extracts **Annual Income** and **Spending Score** as input features.  

### **3. Determine the Optimal Number of Clusters (Elbow Method)**  
- Runs **K-Means clustering** with different cluster numbers (1 to 10).  
- Uses **Within-Cluster Sum of Squares (WCSS)** to find the best number of clusters.  
- Plots an **Elbow Graph** to visualize the optimal number of clusters.  

### **4. Apply K-Means Clustering**  
- Uses **5 clusters** (determined from the Elbow Method).  
- Predicts customer segments based on spending behavior.  

### **5. Visualize the Clusters**  
- Plots customer groups in different colors.  
- Highlights **cluster centroids** for better interpretation.  

---

## **Code Breakdown**  

### **1. Import Required Libraries**  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```
- `numpy` & `pandas`: Data handling.  
- `matplotlib.pyplot` & `seaborn`: Data visualization.  
- `KMeans` from `sklearn.cluster`: Clustering algorithm.  

---

### **2. Load and Inspect the Dataset**  
```python
customer_data = pd.read_csv('Mall_Customers.csv')

customer_data.head()
customer_data.shape
customer_data.info()
customer_data.isnull().sum()
```
- Reads the dataset.  
- Checks **missing values, data types, and structure**.  

---

### **3. Select Features for Clustering**  
```python
X = customer_data.iloc[:, [3,4]].values
print(X)
```
- Selects **Annual Income** and **Spending Score** for clustering.  

---

### **4. Determine the Optimal Number of Clusters (Elbow Method)**  
```python
wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```
- Runs K-Means for **1 to 10 clusters**.  
- Computes **Within-Cluster Sum of Squares (WCSS)** for each value of K.  
- Plots an **Elbow Graph** to find the optimal number of clusters.  

---

### **5. Apply K-Means Clustering with 5 Clusters**  
```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)
print(Y)
```
- Sets **K = 5** based on the Elbow Graph.  
- Predicts **cluster assignments** for customers.  

---

### **6. Visualize the Customer Segments**  
```python
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
```
- Plots **customer clusters** in different colors.  
- Highlights **centroids** of each cluster.  

---

## **Results & Interpretation**  
- Customers are grouped into **five segments** based on income and spending behavior.  
- Businesses can use these insights to **target different customer groups** for personalized marketing.  
