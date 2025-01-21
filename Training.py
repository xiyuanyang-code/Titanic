# %%
import json
import pandas as pd
import numpy as np

# %%

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

train_df['Parch'] = train_df['Parch'].astype(int)
train_df['SibSp'] = train_df['SibSp'].astype(int)
train_df['Pclass'] = train_df['Pclass'].astype(int)


test_df['Age'] = test_df['Age'].astype(float)
test_df['Fare'] = test_df['Fare'].astype(float)

test_df['Parch'] = test_df['Parch'].astype(int)
test_df['SibSp'] = test_df['SibSp'].astype(int)
test_df['Pclass'] = test_df['Pclass'].astype(int)


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

# %%
# 从姓名中提取称呼（如 Mr, Miss, Mrs）
train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# 将性别转换为数值
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# 创建家庭大小特征
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

# 将登船港口转换为数值
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# %%
# 选择特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
X_train = train_df[features]
y_train = train_df['Survived']

X_test = test_df[features]

# %%
print(X_train.dtypes)

# %%
from sklearn.ensemble import RandomForestClassifier

# 初始化模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix

# 在训练集上进行预测
y_train_pred = model.predict(X_train)

# 计算准确率
accuracy = accuracy_score(y_train, y_train_pred)
print(f"训练集准确率: {accuracy:.2f}")

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_train, y_train_pred)
print("混淆矩阵:")
print(conf_matrix)

# %%
# 对测试集进行预测
test_predictions = model.predict(X_test)

# 将预测结果保存到 CSV 文件
output = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})
output.to_csv('submission.csv', index=False)


