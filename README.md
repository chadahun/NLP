# K-DIGITAL NLP Project
## 사용한 라이브러리: Selenium / Pandas / Numpy / Scikit-learn / Gensim / WordCloud / SpaCy / Matplotlib / Nltk

### 개발 공정
#### 1. 잡코리아에서 합격 자소서 크롤링(합격자소서에 전문가 평이 1점~5점까지 있고 1/2점이 안좋은 자소서, 4/5점이 좋은 자소서라는 전제) 후 파파고에 한글 자소서를 넣어서 영어로 번역한것을 크롤링
     애로사항: 1~5점까지의 자소서들의 데이터 양이 일괄적이지가 않음 / 홈페이지 이용약관을 고려하지 않고 크롤링 유의사항 고려하지 않음 / 지금껏 배운 기술로는 한글의 전처리가 아쉬워서 영어로 번역
#### 2. EDA, 워드클라우드 
     애로사항: 처음에는 1점부터 5점까지 모든 평점이 맥여진 자소서들을 가지고 해봤지만 이렇다 할 차이가 크게 없었다 / 분류 모델을 사용하기 위해서 1~5점까지는 분류가 잘 되지 않을것으로 판단, 1점과 5점만으로 분류를 하려했지만 데이터 양이 너무 적으므로 1/2점, 4,5점으로 나눔
#### 3. 벡터화(Doc2Vec, SpaCy) 모델링
     애로사항: SpaCy 모델로 벡터화 후 성능을 확인해 봤지만 성능이 원하는 만큼 나오지 않았을 뿐더러 SpaCy 모델 사용법 미흡 / 
#### 4. 모델링(DecisionTreeClassifier, RandomForestClassifier) 및 하이퍼 파라미터 튜닝(GridSearchCV, RandomSearchCV)
     애로사항: 모델간에 성능이 크게 차이나지 않음 / 처음엔 optuna를 사용하려 했지만 데이터가 많은 편이 아니라서 GridSearch나 RandomSearch 사용
#### 5. 모델링(Linear Regression, DecisionTreeRegressor, RandomForestRegressor)
     애로사항: 선형회귀 모델의 성능을 확인했을 때 RMSE는 괜찮게 나왔지만 R^2계수가 아주 낮게 나옴 / 회귀 모델은 사용할 만한 데이터가 아닌것으로 판단하여 성능만 확인 
