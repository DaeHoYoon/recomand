#%%
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
#%%
# 데이터 정의
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size = 0.25, random_state=0)
# %%
# 잠재요인 협업필터링 수행
# Process : 알고리즘 정의 > 학습 > 예측(test: 전체 데이터셋, predict: 개별 사용자)

# 알고리즘 정의 및 학습
model = SVD()
model.fit(trainset) # 학습
# %%
# 전체 예측 : test
predictions = model.test(testset) # 테스트 전체에 대한 추천영화 평점 데이터 반환
print("prediction type:", type(predictions), "size:", len(predictions))
print("prediction 결과의 최초 5개 추출")
print(predictions[:5])
# %%
# 반환된 객체 값 가져오기
[(pred.uid, pred.iid, pred.est) for pred in predictions[:3]] 
# %%
# 개별 예측 : predict(uid(문자열), iid(문자열), r_ui(선택사항))
uid = str(196) # 196번째 사용자
iid = str(302) # 302번 아이템

pred = model.predict(uid, iid)
print(pred)
# %%
# 정확도 평가 : 실제 평점(r_ui)과 예측 평점(est)과의 차이를 평가
accuracy.rmse(predictions)
# %%
# 주요 API
# 1. Dataset.load_builtin : 데이터 내려받기
# 2. Dataset.load_from_file(file_path, reader): os 파일에서 데이터를 로딩
#   - file_path: os 파일명
#   - reader : 파일의 포맷
# 3. Dataset.load_from_df(df, reader): 판다스 데이터프레임에서 데이터를 로딩
#   - df : 데이터프레임(userId, itemId, rating 순으로 컬럼이 정해져 있어야함)
#   - reader : 파일의 포맷

#%%
# os파일 데이터 surprise 데이터셋으로 로딩
import pandas as pd

# 데이터 불러오기
ratings = pd.read_csv(r'.\data\grouples_ratings.csv')

# 컬럼명 제거 후 데이터 저장
ratings.to_csv(r'C:\BIG_DATA\recomand\recomand\Movie 3 reco\data\rating_dh', index=False, header=False)
# %%
# os파일 불러오기
# os파일을 불러오기 위해서는 불러오려는 기존 데이터프레임의 컬럼, 평점 관련 정보를 입력해야함

from surprise import Reader
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5,5))
data = Dataset.load_from_file(r'.\data\rating_dh', reader=reader)

trainset, testset = train_test_split(data, test_size=0.25, random_state=0)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)
accuracy.rmse(predictions)
# %%
# svd parameter
# n_factor = 잠재요인 개수
# n_epochs = SGD 수행 시 반복 횟수
# biased = 베이스라인 사용자 편향 적용 여부, default = True
# %%
# Baseline 평점: 사용자의 성향을 반영하여 평점 계산
#    - Baseline 평점 = (전체 평균 평점) + (사용자 편향 점수) + (아이템 편향 점수)
#    - 전체 평균 평점 : 모든 사용자의 모든 아이템에 대한 평점을 평균한 값
#    - 사용자 편향 점수 : (사용자별 아이템 평점 평균) - (전체 평균 평점)
#    - 아이템 편향 점수 : (아이템별 평균 평점) - (전체 평균 평점)
# %%
# Surprise 교차검증 및 하이퍼파라미터 튜닝 cross_validate
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import Reader
# %%
ratings = pd.read_csv(r'.\data\grouples_ratings.csv')
reader = Reader(rating_scale=(0.5,5))
data = Dataset.load_from_df(ratings[['userId', 'movieId','rating']], reader=reader)

model = SVD()
cvDict = cross_validate(algo=model, data=data, measures=["RMSE","MSE"], cv=5, verbose=False)
cvDict_df = pd.DataFrame(cvDict)
cvDict_df = cvDict_df.T # T=transpose()
cvDict_df.columns = ["Fold1","Fold2","Fold3","Fold4","Fold5"]
cvDict_df
# %%
from surprise.model_selection import GridSearchCV

param_grid = {"n_epochs":[20,40,60], "n_factors":[5,100,200]}

gridSearch = GridSearchCV(algo_class=SVD, param_grid=param_grid, measures=['RMSE','MAE'], cv=5)
gridSearch.fit(data)
# %%
gridSearch.best_params
gridSearchdf = pd.DataFrame(gridSearch.cv_results)
gridSearchdf = gridSearchdf[['params','mean_test_rmse','rank_test_rmse','mean_test_mae','rank_test_mae']]
gridSearchdf.head()
# %%
