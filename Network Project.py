#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


df = pd.read_csv(r"C:\Users\User\Downloads\archive (5)\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")


# In[4]:


df


# In[5]:


missing_values = df.isnull().sum()
missing_values


# In[6]:


pd.set_option('display.max_rows', None)  #displays all rows and columns
pd.set_option('display.max_columns', None)

missing_percentage = df.isnull().mean() * 100 
missing_percentage


# In[7]:


missing_percentage = df.isnull().mean() * 100   #Display Only Columns with Missing Values
missing_percentage[missing_percentage > 0] 


# In[8]:


#Fill Missing Values with Mean/Median:

df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean(), inplace=True)


# In[9]:


missing_percentage = df.isnull().mean() * 100   #Display Only Columns with Missing Values(we can see no missing data)
missing_percentage[missing_percentage > 0] 


# In[10]:


pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)
missing_values = df.isnull().sum()
missing_values


# In[11]:


df.info()


# In[12]:


# Selecting relevant features for anomaly detection (excluding string columns like IP addresses)
# Keeping columns with numeric values related to traffic analysis
selected_columns = [
    ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 
    'Total Length of Fwd Packets', ' Total Length of Bwd Packets', 
    ' Fwd Packet Length Max', 'Bwd Packet Length Max', 'Flow Bytes/s', ' Flow Packets/s',
    ' Flow IAT Mean', ' Bwd IAT Mean', ' Flow IAT Std', ' Bwd IAT Std',
    'Active Mean', 'Idle Mean', ' Active Max', ' Idle Max', ' Labels'
]

# Creating a new DataFrame with the selected columns
df_filtered = df[selected_columns]

# Encoding the Labels column: BENIGN = 0, Web Attack ï¿½ Brute Force = 1 (or any other appropriate encoding)
df_filtered[' Labels'] = df_filtered[' Labels'].map({'BENIGN': 0, 'Web Attack': 1})

# Display the cleaned and encoded dataset
df_filtered.head()


# In[13]:


from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[14]:


import numpy as np

# Replace infinite values with NaN
df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values (or you can choose to fill them with the mean/median)
df_filtered.dropna(inplace=True)

# Alternatively, you can scale down the features to avoid very large numbers
from sklearn.preprocessing import StandardScaler

# Separating features and labels again after cleaning
X = df_filtered.drop(columns=[' Labels'])
y = df_filtered[' Labels']

# Scaling the features to handle large values (normalizing the data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Re-training the model with the cleaned and scaled data
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Converting predictions to match the label encoding (1 = Web Attack ï¿½ Brute Force, 0 = BENIGN)
y_pred = [1 if pred == -1 else 0 for pred in y_pred]

# Evaluating the model
print(classification_report(y_test, y_pred, target_names=['BENIGN', 'Web Attack']))


# In[15]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['BENIGN', 'Web Attack'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()



# In[16]:


fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


# In[17]:


# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming X_test is a NumPy array, convert it to a DataFrame
X_test_df = pd.DataFrame(X_test, columns=X.columns)  # Using the column names from X

# Assuming y_pred is a NumPy array, convert it to a Series
y_pred_series = pd.Series(y_pred, name='Anomaly')

# Add the predictions (Anomaly) to the test DataFrame
X_test_df['Anomaly'] = y_pred_series

# Plot the bar chart for anomaly counts
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Anomaly', data=X_test_df, palette={0: 'blue', 1: 'red'})

# Add the counts on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

# Add titles and labels
plt.title('Count of Benign vs Malicious Traffic', fontsize=16)
plt.xlabel('Anomaly (0 = Benign, 1 = Web Attacks)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Show the plot
plt.show()


# In[ ]:


import pandas as pd
data = pd.read_csv(r"C:\Users\User\Downloads\archive (5)\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
unique_labels = data[' Labels'].unique()
print(unique_labels)


# In[ ]:




