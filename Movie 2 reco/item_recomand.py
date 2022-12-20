#%%
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

#%%
movies = pd.read_csv(r'.\data\grouplens_movies.csv')
ratings = pd.read_csv(r'.\data\grouples_ratings.csv')
print(movies.shape)
print(ratings.shape)

# 공통된 컬럼 확인
# print(movies.columns)
# print(ratings.columns)
movie_col = list(movies.columns)
rate_col = list(ratings.columns)

samecol = [col for col in movie_col if col in rate_col]
samecol

# 공통된 컬럼 기준으로 merge함
total_df = pd.merge(movies, ratings, how = 'inner', on = 'movieId')
total_df
print(total_df.shape)

itemMt = pd.pivot_table(data = total_df, index='userId', columns = 'title', values ='rating')
print(itemMt.shape)
itemMt_f = itemMt.fillna(0)
print(itemMt.head())

# 아이템 기반 추천은 아이템이 row, 사용자가 컬럼에 있어야함
itemMt_F = itemMt_f.transpose()
itemMt_F.head()
print(itemMt_F.shape)
# %%
item_sim = cosine_similarity(itemMt_F, itemMt_F)
item_sim.argsort()[:,::-1]
print(item_sim.shape)
# %%
df = pd.DataFrame(item_sim, index = itemMt_F.index, columns = itemMt_F.index)
df
# %%
# Inception과 유사한 상위 10개 아이템을 추천
df['Inception (2010)'].sort_values(ascending=False)[:10]
# %%
print("순서 : 최종 df > 유사도 df > 최종 df * 유사도 df > 새로운 df")
# %%
# 예측 평점 계산
# 최종 df와 유사도 df의 행렬곱
new_df = itemMt_f.values.dot(df)
# new_df.values / np.abs(df.values.sum())
p_df = itemMt_f.dot(df) / np.array([np.abs(df).sum(axis=1)])
# %%
# 최종 df, 예측 평점 df의 mse 계산
from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    
    actual1 = actual[actual.nonzero()]
    pred1 = pred[actual.nonzero()]

    return mean_squared_error(pred1, actual1)

# %%
MSE = get_mse(itemMt_f.values, p_df.values)
print(MSE)
# %%
# 특정 영화와 유사도가 높은 영화들로 에측 평점 구하기
pred = np.zeros(itemMt_f.shape)
print(f"pred shape: {pred.shape}")
for col in range(itemMt_f.shape[1]):
    top20sim = np.argsort(df.values[:,col])[:-21:-1]
    for row in range(itemMt_f.shape[0]):
        pred[row, col] = df.values[col,:][top20sim].dot(itemMt_f.values[row,:][top20sim])
        pred[row, col] /= np.abs(df.values[col, :][top20sim]).sum()

# %%
pred_df = pd.DataFrame(pred, index = itemMt_f.index, columns = itemMt_f.columns)
pred_df.shape
# %%
MSE1 = get_mse(itemMt_f.values, pred_df.values)
print(MSE1)
# %%
# 9번 사용자가 어떤 영화를 좋아하는지 확인
user9 = itemMt_f.loc[9,:]
print(user9.sort_values(ascending=False)[:10])

# 본 영화 제외
idSeries = itemMt_f.loc[9,:]
alrSeen = idSeries[idSeries > 0].index.tolist() # 이미 본 영화들 리스트
allMovies = list(itemMt_f)
unSeen = [i for i in allMovies if i not in alrSeen]
unSeen
# %%
# 9번한테 추천
print(pred_df.loc[9,unSeen].sort_values(ascending=False)[:10])
# %%
itemRecomDF = pd.DataFrame(pred_df.loc[9,unSeen].sort_values(ascending=False)[:10])
itemRecomDF.rename(columns = {9:'pred'}, inplace = True)
itemRecomDF
# %%
