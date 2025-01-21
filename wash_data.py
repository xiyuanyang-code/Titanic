import json
import pandas as pd
import numpy as np

'''Data washing'''

# 加载训练集 JSON 数据
with open('train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 加载测试集 JSON 数据
with open('test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)


# 将 JSON 数据转换为 DataFrame
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# 将字符串类型的数值转换为数值类型
train_df['Age'] = train_df['Age'].replace('', np.nan)
train_df['Fare'] = train_df['Fare'].replace('', np.nan)
test_df['Age'] = test_df['Age'].replace('', np.nan)
test_df['Fare'] = test_df['Fare'].replace('', np.nan)

train_df['Age'] = train_df['Age'].astype(float)
train_df['Fare'] = train_df['Fare'].astype(float)

test_df['Age'] = test_df['Age'].astype(float)
test_df['Fare'] = test_df['Fare'].astype(float)


# 处理缺失值
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df['Cabin'] = train_df['Cabin'].fillna('Unknown')

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Cabin'] = test_df['Cabin'].fillna('Unknown')

# check
# 检查训练集的缺失值
print("The loss of train set:")
print(train_df.isnull().sum())

# 检查测试集的缺失值
print("\nThe loss of test set:")
print(test_df.isnull().sum())

# 选择特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']
X_train = train_df[features]
y_train = train_df['Survived']

X_test = test_df[features]