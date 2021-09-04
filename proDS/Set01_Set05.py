# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 12:53:16 2021

@author: Administrator
"""

# %%

# =============================================================================
# =============================================================================
# # 문제 01 유형(DataSet_01.csv 이용)
#
# 구분자 : comma(“,”), 4,572 Rows, 5 Columns, UTF-8 인코딩
#
# 글로벌 전자제품 제조회사에서 효과적인 마케팅 방법을 찾기
# 위해서 채널별 마케팅 예산과 매출금액과의 관계를 분석하고자
# 한다.
# 컬 럼 / 정 의  /   Type
# TV   /     TV 마케팅 예산 (억원)  /   Double
# Radio / 라디오 마케팅 예산 (억원)  /   Double
# Social_Media / 소셜미디어 마케팅 예산 (억원)  / Double
# Influencer / 인플루언서 마케팅
# (인플루언서의 영향력 크기에 따라 Mega / Macro / Micro /
# Nano) / String

# SALES / 매출액 / Double
# =============================================================================
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error  # 결정계수
from scipy.stats import ttest_rel
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import scipy.stats as sc
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.api import OLS, add_constant
from sklearn.linear_model import LinearRegression
import pandas as pd

dat = pd.read_csv('Dataset_01.csv')


# %%

# =============================================================================
# 1. 데이터 세트 내에 총 결측값의 개수는 몇 개인가? (답안 예시) 23
# =============================================================================

# 열별 결측치 수 확인 (정답)
dat.isna().sum().sum()

# (비교)
dat.isna().sum(axis=1)  # 행별 결측치 수 확인


# %%

# =============================================================================
# 2. TV, Radio, Social Media 등 세 가지 다른 마케팅 채널의 예산과 매출액과의 상관분석을
# 통하여 각 채널이 매출에 어느 정도 연관이 있는지 알아보고자 한다.
# - 매출액과 가장 강한 상관관계를 가지고 있는 채널의 상관계수를 소수점 5번째
# 자리에서 반올림하여 소수점 넷째 자리까지 기술하시오. (답안 예시) 0.1234
# =============================================================================

dat.columns

Q2_abs = dat[['TV', 'Radio', 'Social_Media', 'Sales']].corr().abs()

# 정답
Q2_ans = Q2_abs['Sales'].sort_values(ascending=False)[1]

# (비교)
Q2_abs['Sales'].nlargest(2)[1]


# %%

# =============================================================================
# 3. 매출액을 종속변수, TV, Radio, Social Media의 예산을 독립변수로 하여 회귀분석을
# 수행하였을 때, 세 개의 독립변수의 회귀계수를 큰 것에서부터 작은 것 순으로
# 기술하시오.
# - 분석 시 결측치가 포함된 행은 제거한 후 진행하며, 회귀계수는 소수점 넷째 자리
# 이하는 버리고 소수점 셋째 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================


dat1 = dat.dropna()

lm = LinearRegression(fit_intercept=True)
lm.fit(dat1.drop(columns=['Influencer', 'Sales']), dat1.Sales)

dir(lm)
lm.coef_  # 정답

x = dat1.drop(columns=['Influencer', 'Sales'])
xx = add_constant(x)
y = dat1.Sales

ols1 = OLS(y, x).fit()  # 상수항 미포함
ols1.summary()

# 의미있는 변수 : P > |t|가 0.05보다 작은 것의 개수

ols2 = OLS(y, xx).fit()  # 상수항 포함
ols2.summary()
# 1. 선형성 가정
# R-squared : 결정계수
# Adj R-squared : 변수의 증가에 따른 모델의 증가량에 따라 수정된 결정계수
# 상수항 미포함의 경우 올림없는 본래의 값을 보여주는데 상수항 포함의 경우 올림을 하여 보여줌
var_list = dat1.columns[:-1]

form = 'Sales~'+'+'.join(var_list)

ols3 = ols('Sales~TV+Radio+Social_Media', data=dat1).fit()
ols3.summary()

ols4 = ols(form, data=dat1).fit()
ols4.summary()

# %%

# =============================================================================
# =============================================================================
# # 문제 02 유형(DataSet_02.csv 이용)
# 구분자 : comma(“,”), 200 Rows, 6 Columns, UTF-8 인코딩

# 환자의 상태와 그에 따라 처방된 약에 대한 정보를 분석하고자한다
#
# 컬 럼 / 정 의  / Type
# Age  / 연령 / Integer
# Sex / 성별 / String
# BP / 혈압 레벨 / String
# Cholesterol / 콜레스테롤 레벨 /  String
# Na_to_k / 혈액 내 칼륨에 대비한 나트륨 비율 / Double
# Drug / Drug Type / String
# =============================================================================
# =============================================================================

dataset2 = pd.read_csv('Dataset_02.csv')
dataset2.columns
dataset2.dtypes
dataset2.shape


# %%

# =============================================================================
# 1.해당 데이터에 대한 EDA를 수행하고, 여성으로 혈압이 High, Cholesterol이 Normal인
# 환자의 전체에 대비한 비율이 얼마인지 소수점 네 번째 자리에서 반올림하여 소수점 셋째
# 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================
Q1 = pd.crosstab(index=[dataset2.Sex, dataset2.BP],
                 columns=dataset2.Cholesterol, normalize=True)
# 0.105
Q1 = pd.pivot_table(data=dataset2, index=['Sex', 'BP'], columns=[
                    'Cholesterol'], values=['Drug'], aggfunc='count')/len(dataset2)
Q1

# %%

# =============================================================================
# 2. Age, Sex, BP, Cholesterol 및 Na_to_k 값이 Drug 타입에 영향을 미치는지 확인하기
# 위하여 아래와 같이 데이터를 변환하고 분석을 수행하시오.
# - Age_gr 컬럼을 만들고, Age가 20 미만은 ‘10’, 20부터 30 미만은 ‘20’, 30부터 40 미만은
# ‘30’, 40부터 50 미만은 ‘40’, 50부터 60 미만은 ‘50’, 60이상은 ‘60’으로 변환하시오.
# - Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30
# 초과는 ‘Lv4’로 변환하시오.
# - Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정을
# 수행하시오.
# - 검정 수행 결과, Drug 타입과 연관성이 있는 변수는 몇 개인가? 연관성이 있는 변수
# 가운데 가장 큰 p-value를 찾아 소수점 여섯 번째 자리 이하는 버리고 소수점 다섯
# 번째 자리까지 기술하시오.
# (답안 예시) 3, 1.23456
# =============================================================================


Q2 = dataset2.copy()

Q2['Age_gr'] = np.where(Q2.Age < 20, '10',
                        np.where(Q2.Age < 30, '20',
                                 np.where(Q2.Age < 40, '30',
                                          np.where(Q2.Age < 50, '40',
                                                   np.where(Q2.Age < 60, '50', '60')))))

age_gr = pd.cut(Q2.Age, [0, 20, 30, 40, 50, 60, Q2.Age.max()+1],
                right=False,
                labels=['10', '20', '30', '40', '50', '60'])

# - Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30
# 초과는 ‘Lv4’로 변환하시오.

Q2['Na_k_gr'] = np.where(Q2['Na_to_K'] <= 10, 'Lv1',
                         np.where(Q2['Na_to_K'] <= 20, 'Lv2',
                                  np.where(Q2['Na_to_K'] <= 30, 'Lv3', 'Lv4')))

# - Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정을
# 수행하시오.


# 입력값이 빈도 테이블로 들어가도록 변경해야함
Age_t = pd.crosstab(Q2.Age_gr, Q2.Drug)
out_age_t = sc.chi2_contingency(Age_t)
print(out_age_t[1])  # 정답

# - 검정 수행 결과, Drug 타입과 연관성이 있는 변수는 몇 개인가? 연관성이 있는 변수
# 가운데 가장 큰 p-value를 찾아 소수점 여섯 번째 자리 이하는 버리고 소수점 다섯
# 번째 자리까지 기술하시오.
Q2_temp = []
for i in ['Age_gr', 'Sex', 'BP', 'Cholesterol', 'Na_k_gr']:
    temp = pd.crosstab(Q2[i], Q2.Drug)
    out_age_t = sc.chi2_contingency(temp)
    print(i, out_age_t[1])
    Q2_temp = Q2_temp + [[i, out_age_t[1]]]
Q2_temp = pd.DataFrame(Q2_temp)
Q2_temp.sort_values(by=1)[1][1]

Q2_temp.columns = ['var', 'pvalues']
Q2_temp2 = Q2_temp[Q2_temp.pvalues < 0.05]
len(Q2_temp2)

Q2_temp2.sort_values(by='pvalues').tail(1)  # 정답
# %%

# =============================================================================
# 3.Sex, BP, Cholesterol 등 세 개의 변수를 다음과 같이 변환하고 의사결정나무를 이용한
# 분석을 수행하시오.
# - Sex는 M을 0, F를 1로 변환하여 Sex_cd 변수 생성
# - BP는 LOW는 0, NORMAL은 1 그리고 HIGH는 2로 변환하여 BP_cd 변수 생성
# - Cholesterol은 NORMAL은 0, HIGH는 1로 변환하여 Ch_cd 생성
# - Age, Na_to_k, Sex_cd, BP_cd, Ch_cd를 Feature로, Drug을 Label로 하여 의사결정나무를
# 수행하고 Root Node의 split feature와 split value를 기술하시오.
# 이 때 split value는 소수점 셋째 자리까지 반올림하여 기술하시오. (답안 예시) Age,
# 12.345
# =============================================================================

Q3 = dataset2.copy()

Q3['Sex_cd'] = np.where(Q3['Sex'] == 'M', 0, 1)
Q3['BP_cd'] = np.where(Q3['BP'] == 'LOW', 0,
                       np.where(Q3['BP'] == 'NORMAL', 1, 2))
Q3['Ch_cd'] = np.where(Q3['Cholesterol'] == 'NORMAL', 0, 1)

x_var = ['Age', 'Na_to_K', 'Sex_cd', 'BP_cd', 'Ch_cd']

dt = DecisionTreeClassifier().fit(Q3[x_var], Q3.Drug)

plot_tree(dt, feature_names=x_var, class_names=Q3.Drug.unique())
export_text(dt, feature_names=x_var, decimals=3)  # Na_to_K, 14.83
# %%

# =============================================================================
# =============================================================================
# # 문제 03 유형(DataSet_03.csv 이용)
#
# 구분자 : comma(“,”), 5,001 Rows, 8 Columns, UTF-8 인코딩
# 안경 체인을 운영하고 있는 한 회사에서 고객 사진을 바탕으로 안경의 사이즈를
# 맞춤 제작하는 비즈니스를 기획하고 있다. 우선 데이터만으로 고객의 성별을
# 파악하는 것이 가능할 지를 연구하고자 한다.
#
# 컬 럼 / 정 의 / Type
# long_hair / 머리카락 길이 (0 – 길지 않은 경우 / 1 – 긴
# 경우) / Integer
# forehead_width_cm / 이마의 폭 (cm) / Double
# forehead_height_cm / 이마의 높이 (cm) / Double
# nose_wide / 코의 넓이 (0 – 넓지 않은 경우 / 1 – 넓은 경우) / Integer
# nose_long / 코의 길이 (0 – 길지 않은 경우 / 1 – 긴 경우) / Integer
# lips_thin / 입술이 얇은지 여부 0 – 얇지 않은 경우 / 1 –
# 얇은 경우) / Integer
# distance_nose_to_lip_long / 인중의 길이(0 – 인중이 짧은 경우 / 1 – 인중이
# 긴 경우) / Integer
# gender / 성별 (Female / Male) / String
# =============================================================================
# =============================================================================


# %%

# =============================================================================
# 1.이마의 폭(forehead_width_cm)과 높이(forehead_height_cm) 사이의
# 비율(forehead_ratio)에 대해서 평균으로부터 3 표준편차 밖의 경우를 이상치로
# 정의할 때, 이상치에 해당하는 데이터는 몇 개인가? (답안 예시) 10
# =============================================================================
dt3 = pd.read_csv('DataSet_03.csv')

dt3['forehead_ratio'] = dt3.forehead_width_cm/dt3.forehead_height_cm
m1 = dt3['forehead_ratio'].mean()
sd1 = dt3['forehead_ratio'].std()

LL = m1 - (3*sd1)
UU = m1 + (3*sd1)

Q1_out = dt3[(dt3['forehead_ratio'] < LL) | (dt3['forehead_ratio'] > UU)]
Q1_ans = len(Q1_out)  # 정답
# %%

# =============================================================================
# 2.성별에 따라 forehead_ratio 평균에 차이가 있는지 적절한 통계 검정을 수행하시오.
# - 검정은 이분산을 가정하고 수행한다. <- 독립이 아닐 경우 분산을 고려하지 않음!
# - 검정통계량의 추정치는 절대값을 취한 후 소수점 셋째 자리까지 반올림하여
# 기술하시오.
# - 신뢰수준 99%에서 양측 검정을 수행하고 결과는 귀무가설 기각의 경우 Y로, 그렇지
# 않을 경우 N으로 답하시오. (답안 예시) 1.234, Y
# =============================================================================

# 독립인 두 집단 간의 평균 차이 검정

ttest_ind(dt3.forehead_ratio[dt3.gender == 'Male'],
          dt3.forehead_ratio[dt3.gender == 'Female'],
          equal_var=False)
# Ttest_indResult(statistic=2.9994984197511543, pvalue=0.0027186702390657176)
# pvalue(의사결정용)가 0.01보다 작기 때문에 귀무가설이 기각된다.
# 2.999(검정 통계량),Y
# %%

# =============================================================================
# 3.주어진 데이터를 사용하여 성별을 구분할 수 있는지 로지스틱 회귀분석을 적용하여
# 알아 보고자 한다.
# - 데이터를 7대 3으로 나누어 각각 Train과 Test set로 사용한다. 이 때 seed는 123으로
# 한다.
# - 원 데이터에 있는 7개의 변수만 Feature로 사용하고 gender를 label로 사용한다.
# (forehead_ratio는 사용하지 않음)
# - 로지스틱 회귀분석 예측 함수와 Test dataset를 사용하여 예측을 수행하고 정확도를
# 평가한다. 이 때 임계값은 0.5를 사용한다.
# - Male의 Precision 값을 소수점 둘째 자리까지 반올림하여 기술하시오. (답안 예시)
# 0.12
#
#
# (참고)
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# train_test_split 의 random_state = 123
# =============================================================================

# 데이터셋 분리(seed 체크)
# 모델 만들때는 학습용 데이터셋 활용
# 검정 시 테스트용 데이터셋 활용
# logisticregression은 관심 대상을 1로 만들어야함

train, test = train_test_split(dt3, test_size=0.3, random_state=123)

model1 = LogisticRegression().fit(train.drop(columns='gender'),
                                  train.gender)

pred = model1.predict(test.drop(columns='gender'))

model1.predict_proba(test.drop(columns='gender'))

print(classification_report(test.gender, pred))

precision_score(test.gender, pred, pos_label='Male')

# %%

# =============================================================================
# =============================================================================
# # 문제 04 유형(DataSet_04.csv 이용)
#
# 구분자 : comma(“,”), 6,718 Rows, 4 Columns, UTF-8 인코딩

# 한국인의 식생활 변화가 건강에 미치는 영향을 분석하기에 앞서 육류
# 소비량에 대한 분석을 하려고 한다. 확보한 데이터는 세계 각국의 1인당
# 육류 소비량 데이터로 아래와 같은 내용을 담고 있다.

# 컬 럼 / 정 의 / Type
# LOCATION / 국가명 / String
# SUBJECT / 육류 종류 (BEEF / PIG / POULTRY / SHEEP) / String
# TIME / 연도 (1990 ~ 2026) / Integer
# Value / 1인당 육류 소비량 (KG) / Double
# =============================================================================
# =============================================================================

# (참고)
# #1
# import pandas as pd
# import numpy as np
# #2
# from scipy.stats import ttest_rel
# #3
# from sklearn.linear_model import LinearRegression

dt4 = pd.read_csv("DataSet_04.csv")
dt4

# %%

# =============================================================================
# 1.한국인의 1인당 육류 소비량이 해가 갈수록 증가하는 것으로 보여 상관분석을 통하여
# 확인하려고 한다.
# - 데이터 파일로부터 한국 데이터만 추출한다. 한국은 KOR로 표기되어 있다.
# - 년도별 육류 소비량 합계를 구하여 TIME과 Value간의 상관분석을 수행하고
# 상관계수를 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지만 기술하시오.
# (답안 예시) 0.55
# =============================================================================

dt4_kor = dt4[(dt4.LOCATION == 'KOR')]
dt4_t = dt4_kor.groupby(['TIME']).sum()
dt4_t.reset_index(inplace=True)
dt4_t.corr()['Value']['TIME']
# %%

# =============================================================================
# 2. 한국 인근 국가 가운데 식생의 유사성이 상대적으로 높은 일본(JPN)과 비교하여, 연도별
# 소비량에 평균 차이가 있는지 분석하고자 한다.
# - 두 국가의 육류별 소비량을 연도기준으로 비교하는 대응표본 t 검정을 수행하시오.
# - 두 국가 간의 연도별 소비량 차이가 없는 것으로 판단할 수 있는 육류 종류를 모두
# 적으시오. (알파벳 순서) (답안 예시) BEEF, PIG, POULTRY, SHEEP
# =============================================================================

# dt4_q2 = dt4[dt4.LOCATION.isin(['KOR','JPN'])]
# dt4_t = dt4_q2.groupby(['LOCATION', 'SUBJECT', 'TIME']).mean()
# dt4_t.reset_index(inplace=True)
# dt4_t
# ttest_rel(dt4_t.Value[dt4_t.LOCATION == 'JPN'],
#           dt4_t.Value[dt4_t.LOCATION == 'KOR']) # 독립표본


q2 = dt4[dt4.LOCATION.isin(['KOR', 'JPN'])]
q2.columns
q2_out = pd.pivot_table(data=q2, index=['TIME', 'SUBJECT'], columns=[
                        'LOCATION'], values='Value')

q2_out = q2_out.dropna()
q2_out = q2_out.reset_index()

sub_list = q2_out.SUBJECT.unique()

q2_out2 = []

for i in sub_list:
    temp = q2_out[q2_out.SUBJECT == i]
    pvalue = ttest_rel(temp['KOR'], temp['JPN']).pvalue  # 대응표본
    q2_out2 = q2_out2+[[i, pvalue]]
q2_out2 = pd.DataFrame(q2_out2, columns=['sub', 'pvalue'])
q2_out2[q2_out2.pvalue >= 0.05]  # 차이가 없다는 말은 귀무가설을 채택하겠다는 의미이므로
# pvalue가 0.05 이상인 경우 귀무가설 기각하지 않음 즉 채택

# %%

# =============================================================================
# 3.(한국만 포함한 데이터에서) Time을 독립변수로, Value를 종속변수로 하여 육류
# 종류(SUBJECT) 별로 회귀분석을 수행하였을 때, 가장 높은 결정계수를 가진 모델의
# 학습오차 중 MAPE를 반올림하여 소수점 둘째 자리까지 기술하시오. (답안 예시) 21.12
# (MAPE : Mean Absolute Percentage Error, 평균 절대 백분율 오차)
# (MAPE = Σ ( | y - y ̂ | / y ) * 100/n ))
#
# =============================================================================


q3 = q2_out.drop(columns='JPN')

q3['TIME'].shape  # (146,) 1차원
q3[['TIME']].shape  # (146,1) 2차원

q3_out = []

sub_list = q3.SUBJECT.unique()

for i in sub_list:
    temp = q3[q3.SUBJECT == i]
    lm = LinearRegression().fit(temp[['TIME']], temp.KOR)
    # X(입력변수) 2차원 구조로 입력
    r2_score = lm.score(temp[['TIME']], temp.KOR)
    q3_out = q3_out+[[i, r2_score]]

q3_out = pd.DataFrame(q3_out, columns=['sub', 'r2_score'])

temp = q3[q3.SUBJECT == 'POULTRY']
lm2 = LinearRegression().fit(temp[['TIME']], temp.KOR)
pred = lm2.predict(temp[['TIME']])

mape = (((temp.KOR - pred).abs() / temp.KOR).sum() * 100)/len(temp)
mape

# %%
# (기타)
q3_out = []

sub_list = q3.SUBJECT.unique()

for i in sub_list:
    temp = q3[q3.SUBJECT == i]
    globals()['lm_'+str(i)] = LinearRegression().fit(temp[['TIME']], temp.KOR)
    # X(입력변수) 2차원 구조로 입력
    r2_score = eval('lm_'+str(i)).score(temp[['TIME']], temp.KOR)
    q3_out = q3_out+[[i, r2_score]]

q3_out = pd.DataFrame(q3_out, columns=['sub', 'r2_score'])


# %%

# =============================================================================
# =============================================================================
# # 문제 05 유형(DataSet_05.csv 이용)
#
# 구분자 : comma(“,”), 8,068 Rows, 12 Columns, UTF-8 인코딩
#
# A자동차 회사는 신규 진입하는 시장에 기존 모델을 판매하기 위한 마케팅 전략을
# 세우려고 한다. 기존 시장과 고객 특성이 유사하다는 전제 하에 기존 고객을 세분화하여
# 각 그룹의 특징을 파악하고, 이를 이용하여 신규 진입 시장의 마케팅 계획을
# 수립하고자 한다. 다음은 기존 시장 고객에 대한 데이터이다.
#

# 컬 럼 / 정 의 / Type
# ID / 고유 식별자 / Double
# Age / 나이 / Double
# Age_gr / 나이 그룹 (10/20/30/40/50/60/70) / Double
# Gender / 성별 (여성 : 0 / 남성 : 1) / Double
# Work_Experience / 취업 연수 (0 ~ 14) / Double
# Family_Size / 가족 규모 (1 ~ 9) / Double
# Ever_Married / 결혼 여부 (Unknown : 0 / No : 1 / Yes : 2) / Double
# Graduated / 재학 중인지 여부 / Double
# Profession / 직업 (Unknown : 0 / Artist ~ Marketing 등 9개) / Double
# Spending_Score / 소비 점수 (Average : 0 / High : 1 / Low : 2) / Double
# Var_1 / 내용이 알려지지 않은 고객 분류 코드 (0 ~ 7) / Double
# Segmentation / 고객 세분화 결과 (A ~ D) / String
# =============================================================================
# =============================================================================


# (참고)
# 1
# import pandas as pd
# #2
# from scipy.stats import chi2_contingency
# #3
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
# import pydot
dt5 = pd.read_csv('Dataset_05.csv', na_values=["NA", "?", "", " "])
dt5.columns
dt5.dtypes
dt5.shape

# %%

# =============================================================================
# 1.위의 표에 표시된 데이터 타입에 맞도록 전처리를 수행하였을 때, 데이터 파일 내에
# 존재하는 결측값은 모두 몇 개인가? 숫자형 데이터와 문자열 데이터의 결측값을
# 모두 더하여 답하시오.
# (String 타입 변수의 경우 White Space(Blank)를 결측으로 처리한다) (답안 예시) 123
# =============================================================================

dt5.isna().sum().sum()

# %%

# =============================================================================
# 2.이어지는 분석을 위해 결측값을 모두 삭제한다. 그리고, 성별이 세분화(Segmentation)에
# 영향을 미치는지 독립성 검정을 수행한다. 수행 결과, p-value를 반올림하여 소수점
# 넷째 자리까지 쓰고, 귀무가설을 기각하면 Y로, 기각할 수 없으면 N으로 기술하시오.
# (답안 예시) 0.2345, N
# =============================================================================

q2 = dt5.dropna()

q2_tab = pd.crosstab(index=q2.Gender, columns=q2.Segmentation)

q2_out = sc.chi2_contingency(q2_tab)
q2_out[1]
# %%

# =============================================================================
# 3.Segmentation 값이 A 또는 D인 데이터만 사용하여 의사결정 나무 기법으로 분류
# 정확도를
# 측정해 본다.
# - 결측치가 포함된 행은 제거한 후 진행하시오.
# - Train대 Test 7대3으로 데이터를 분리한다. (Seed = 123)
# - Train 데이터를 사용하여 의사결정나무 학습을 수행하고, Test 데이터로 평가를
# 수행한다.
# - 의사결정나무 학습 시, 다음과 같이 설정하시오:
# • Feature: Age_gr, Gender, Work_Experience, Family_Size,
#             Ever_Married, Graduated, Spending_Score
# • Label : Segmentation
# • Parameter : Gini / Max Depth = 7 / Seed = 123
# 이 때 전체 정확도(Accuracy)를 소수점 셋째 자리 이하는 버리고 소수점 둘째자리까지
# 기술하시오.
# (답안 예시) 0.12
# =============================================================================
q3 = q2[q2.Segmentation.isin(['A', 'D'])]


train, test = train_test_split(q3, test_size=0.3, random_state=123)

dt = DecisionTreeClassifier(max_depth=7, random_state=123)

x_var = ['Age_gr', 'Gender', 'Work_Experience', 'Family_Size',
         'Ever_Married', 'Graduated',  'Spending_Score']

dt.fit(train[x_var], train.Segmentation)

pred = dt.predict(test[x_var])

dt.score(test[x_var], test.Segmentation)
