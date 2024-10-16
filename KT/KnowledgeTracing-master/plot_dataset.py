import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

file_path = '../BKT/data/output1.csv'
data = pd.read_csv(file_path)
iid = 8  # 假设我们要分析的是用户ID为1的数据
student_data = data[data['user_id'] == iid]
# student_id = student_data['user_id'].unique()
student_id = student_data['user_id'].unique()
skill_tag = student_data['skill_id'].unique()
problem_id = student_data['problem_id'].unique()
heatmap_data = np.zeros((len(skill_tag), len(problem_id)))
# heatmap_data = pre[student_id]

# print(type(skill_ids))
# 填充预测概率
for i, skill in enumerate(skill_tag):
    for j, pro_id in enumerate(problem_id):
        # 假设这里调用模型预测学生在下一个时间步回答 skill 标签的概率
        predicted_prob = np.random.rand()  # 自行实现该函数
        heatmap_data[i, j] = predicted_prob
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='RdYlGn', annot=True)
plt.xlabel('Time Steps(problem_id)')
plt.ylabel('Skill Tags')
plt.title('Prediction Probabilities Heatmap for User ID-1')
plt.show()