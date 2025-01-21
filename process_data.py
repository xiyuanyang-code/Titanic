import csv
from pathlib import Path
import json

test_data_path = Path('test.csv')
train_data_path = Path('train.csv')

train_lines = train_data_path.read_text().splitlines()
test_lines = test_data_path.read_text().splitlines()
# 将文本字符串按行拆分成列表

train_reader = csv.reader(train_lines)
test_reader = csv.reader(test_lines)

header_row_train = next(train_reader)
header_row_test = next(test_reader)
#得到第一行题头

'''
print(header_row_train)
print(header_row_test)
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
'''

train_data_json = []
test_data_json = []

# 将训练集数据转换为 JSON
for row in train_reader:
    # 将每一行数据与题头组合成字典
    row_dict = {header: value for header, value in zip(header_row_train, row)}
    train_data_json.append(row_dict)

# 将测试集数据转换为 JSON
for row in test_reader:
    # 将每一行数据与题头组合成字典
    row_dict = {header: value for header, value in zip(header_row_test, row)}
    test_data_json.append(row_dict)

# 打印结果
print("训练集 JSON 数据：")
print(train_data_json[:2])

print("\n测试集 JSON 数据：")
print(test_data_json[:2])

# 保存训练集 JSON 数据
with open('train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data_json, f, indent=2)  # indent=2 用于美化输出

# 保存测试集 JSON 数据
with open('test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data_json, f, indent=2)