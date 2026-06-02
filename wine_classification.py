from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 1. 데이터셋 로드
wine = load_wine()

X = wine.data
y = wine.target

print("데이터 크기:", X.shape)
print("타겟 크기:", y.shape)
print("클래스 이름:", wine.target_names)

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=4,
    stratify=y
)

print("훈련 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)
