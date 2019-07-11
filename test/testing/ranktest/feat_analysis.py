import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#df1 = pd.read_csv('features/lax_1448_small_feat23_100k.csv')
#df2 = pd.read_csv('features/nyc_85_feat22_sample1.csv')
df3 = pd.read_csv('features/w_92_small_feat22.csv')
df4 = pd.read_csv('features/w_106_small_feat22.csv')

df_name_pairs = [(df3, 'w_92_small'), (df4, 'w_106_small')]

for df, name in df_name_pairs:
    df = df.drop(columns=['time', 'id', 'vtime', 'access_day', 'access_hr', 'access_min'])
    df = df.iloc[int(.02*df.shape[0]):int(.98*df.shape[0])]

    normalizing_func = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)
    df = normalizing_func(df)

    pca = PCA(n_components=4)
    pca.fit(df)
    print(pca.explained_variance_ratio_)
    comp_arr = pca.components_
    write_df = pd.DataFrame(comp_arr)
    write_df['Var_Ratio'] = pca.explained_variance_ratio_
    write_df.to_csv('features/pca/' + name + '_pca.csv', index=False)
    print('Done with: ' + name)
    
    
#data_trans = np.array(pca.transform(df))

""" fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_trans[:,0], data_trans[:,1], data_trans[:,2], s=1)
ax.set_xlabel('Reuse Distance Statistics')
ax.set_ylabel('Frequency Statistics')
ax.set_zlabel('Request Rate Statistics') """

#plt.show()