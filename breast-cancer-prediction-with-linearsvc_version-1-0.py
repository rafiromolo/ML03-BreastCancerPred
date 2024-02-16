# -- Importing Libraries --
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.model_selection import LearningCurve, FeatureImportances

# -- Reading the Dataset --
df = pd.read_csv('dataset/breast-cancer.csv')

# -- Rename concave point column --
df.rename(columns={'concave points_mean': 'concave_points_mean', 'concave points_se': 'concave_points_se', 'concave points_worst': 'concave_points_worst'}, inplace=True)

# -- Data Pre-processing --
df = df.drop(columns=['id'], axis=1)

x = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

x = MinMaxScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# -- Model Implementation --
model = LinearSVC(random_state=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# -- Model's Accuracy Score --
model_acc = accuracy_score(y_pred, y_test)
print('LinearSVC Accuracy:'+'{:.2f}'.format(model_acc*100)+'\n')
print('Classification Report\n')
print(classification_report(y_test, y_pred))

# -- Model's Performance Evaluation --
print('Performance Evaluation')
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(16,8))
model_matrix = ConfusionMatrix(model, ax=ax1, cmap='YlOrRd', title='LinearSVC Confusion Matrix')
model_matrix.fit(x_train, y_train)
model_matrix.score(x_test, y_test)
model_matrix.finalize()
model_lc = LearningCurve(model, ax=ax2, title='LinearSVC Learning Curve')
model_lc.fit(x_train, y_train)
model_lc.finalize()
plt.tight_layout()

# -- Model's Feature Importances --
viz = FeatureImportances(model, ax=ax3)
viz.fit(x, y)
viz.show()