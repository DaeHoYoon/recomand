#%%
import pandas as pd
import numpy as np
from dataprep.eda import plot, create_report # 시각화
from ast import literal_eval # str을 딕셔너리로 바꿔주는 함수
from sklearn.metrics.pairwise import cosine_similarity # 유사도
import warnings; warnings.filterwarnings('ignore')
#%%
# 가중 평균 함수
def weight_average(df, pct):
    pct = pct
    m = df['vote_count'].quantile(pct)
    c = df['vote_average'].mean()
    v = df['vote_count']
    r = df['vote_average']
    weight_average = (v / (v+m)) * r + (m / (v+m)) * c
    return weight_average

# 가중 평균 기반 추천 시스템 함수
def weight_vote_avg(df, sorted_sim, title = '', num=10):
    title_mv2 = df[df["title"] == title] # 해당 영화보기
    title_mv2_idx = title_mv2.index.values # 해당 영화 인덱스
    print(title_mv2_idx)

    sim_idx2 = sorted_sim[title_mv2_idx, :(num)] # 유사도가 높은 20개
    sim_idx2 = sim_idx2.reshape(-1) # 1차원으로 변경
    sim_idx2 = sim_idx2[sim_idx2 != title_mv2_idx]

    similar_mv2 = df.iloc[sim_idx2]
    return similar_mv2.sort_values(by='weight_average', ascending = False)
# %%
movies = pd.read_csv(r'C:\BIG_DATA\portfolio\recomand\data\tmdb_5000_movies.csv')

# create_report(movies)
# %%
print(f"movies info: {movies.info()}")
# %%
### 20개 컬럼 중에 필요한 컬럼 추출
movies_df = movies[['id','title','genres','vote_average','vote_count','popularity','keywords','overview']]
print(f"movies shape: {movies_df.shape}")
movies_df.head()
# %%
### 장르컬럼 원핫인코딩
# 문자열로 된 value값 딕셔너리로 변환
movies_df['genres'] = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)
# %%
# 장르값만 추출
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i['name'] for i in x])
# %%
# 장르 컬럼 펼치기
genres_list = []
for gen in movies_df['genres']:
    genres_list.extend(gen) # 리스트로 되어있는 장르들 개개별 추출
genres_list = np.unique(genres_list) # 장르의 유니크값 추출
genres_list
# %%
# 원핫 인코딩 매트릭스 만들기
zero_array = np.zeros(shape=(movies_df.shape[0],len(genres_list)))
print(zero_array.shape)
zero_df = pd.DataFrame(zero_array, columns=genres_list)
zero_df
# %%
for idx, genre in enumerate(movies_df['genres']):
    indices = zero_df.columns.get_indexer(genre)
    zero_df.iloc[idx,indices] = 1

zero_df
# %%
### 유사도 구하기
gen_df = zero_df.copy()
gen_sim = cosine_similarity(gen_df, gen_df)
print(f"gen_sim shape: {gen_sim.shape}")
print(gen_sim[0])
# %%
# 정렬
sorted_gen_sim = gen_sim.argsort()[:,::-1]
sorted_gen_sim
# %%
# 1차 추천 : 9번 사람
# title_mv = movies_df[movies_df["title"] == "The Godfather"] # 해당 영화보기
# title_mv_idx = title_mv.index.values # 해당 영화 인덱스
# print(title_mv_idx)

# sim_idx = sorted_gen_sim[title_mv_idx, :10] # 유사도가 높은 10개
# sim_idx = sim_idx.reshape(-1) # 1차원으로 변경

# similar_mv = movies_df.iloc[sim_idx][["title","vote_average"]]
# print(similar_mv)
# %%
### 가중 평점으로 추천
# 가중평점 = (v / (v+m)) * r + (m / (v+m)) * c
# v = 영화별 평점 투표 횟수
# m = 평점 부여를 위한 최소 투표 횟수
# r = 영화별 평균 평점
# c = 전체 영화의 평균 평점

movies_df['weight_average'] = weight_average(movies_df, 0.6)

movies_df[["title","vote_average",'weight_average','vote_count']].sort_values(by="weight_average", ascending=False)
# %%
sim_mv2 = weight_vote_avg(movies_df, sorted_gen_sim, "The Godfather")
sim_mv2 = sim_mv2[["title","weight_average","vote_average"]]
sim_mv2