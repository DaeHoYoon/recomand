#%%
import numpy as np
import pandas as pd

from surprise import Dataset
from surprise.dataset import DatasetAutoFolds
from surprise import SVD
from surprise import accuracy
from surprise import Reader
# %%
### DatasetAutoFolds
# trainset 정의
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0.5,5))
daf = DatasetAutoFolds(ratings_file=r'.\data\rating_dh', reader=reader)
trainset = daf.build_full_trainset()

# 알고리즘 정의
model = SVD(n_epochs = 20, n_factors = 50, random_state=0)
model.fit(trainset)

# testset 정의
movies = pd.read_csv(r'.\data\grouplens_movies.csv')
ratings = pd.read_csv(r'.\data\grouples_ratings.csv')
print(f'movies shape: {movies.shape}')
print(f'ratings shape: {ratings.shape}')

# 안 본 영화 id 추출 (9번이 안 본 영화)
def getUnseenMovie(df, userid):
    seenMovies = df[df["userId"]==userid]["movieId"].tolist() # 유저가 평점을 매긴 영화를 본 영화로 취급
    allMovies = movies["movieId"].tolist() # 전체 영화
    UnseenMovies = [movie for movie in allMovies if movie not in seenMovies] # 안 본 영화
    print(f"평점을 매긴 영화 수: {len(seenMovies)}, \n전체 영화 수: {len(allMovies)}, \n추천해야 할 영화 수: {len(UnseenMovies)}")
    return UnseenMovies
# %%
# 9번 사용자가 보지 않은 영화들의 id
UnseenMovies = getUnseenMovie(ratings, 9)
# %%
# 9번 사용자에게 보지 않은 영화들 중 10개를 추천해주기
# 예측 > predictions 정렬 > TopN개의 predictions를 추출 > TopN개의 id, 제목, 예측 평점 데이터 프레임 생성
def surpriseMoviereco(model, userid, UnseenMovies):
    predictions = [model.predict(str(userid), str(movieId)) for movieId in UnseenMovies]
    def sort_est(pred):
        return pred.est
    predictions.sort(key=sort_est, reverse=True) # key값엔 리스트 안에 있는 surprise 데이터를 하나씩 반환해줄 수 있는 함수가 필요
    top10pred = predictions[:10]
    print(top10pred)
    # top 10으로 추출된 영화의 정보 추출(movieId, est, title)
    topMovieIds = [int(pred.iid) for pred in top10pred]
    topMovieEsts = [pred.est for pred in top10pred]
    topMovieTitles = movies[movies["movieId"].isin(topMovieIds)]['title']

    topMoviePreds = [(id, title, est) for id, title, est in zip(topMovieIds, topMovieTitles, topMovieEsts)]

    topMoviedf = pd.DataFrame(topMoviePreds, columns = ["movieId", "title", "pred"])
    return topMoviedf
# %%
surpriseMoviereco(model, 9, UnseenMovies)
# %%
