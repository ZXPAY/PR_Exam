
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[3.98,1,2.5],
              [1.2,1.78,3.4],
              [8.9,8.8,7.9],
              [9.8,8.7,9.01],
              [1.22,0.6,2.1],
              [7.8,9.5,8.45]],dtype=np.float32)
y = np.array([1,1,2,2,1,2],dtype=np.float32)
test_X=np.array([[5,6,7],[1,1,1],[0.8,9,3.5]],dtype=np.float32)
print(X.shape)

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
print(X)
pca = PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

print(pca.explained_variance_ratio_)    # 共變異比率，可看出其中的特徵，哪個比較重要)





