import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
print("Loading Titanic Dataset...")
df = pd.read_csv(url)


print("\n--- First 5 Rows of Dataset ---")
print(df.head())


print("\n--- Missing Values Summary ---")
print(df.isnull().sum())


df['Age'].fillna(df['Age'].median(), inplace=True)


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


df['Survival_Label'] = df['Survived'].map({0: 'Did Not Survive', 1: 'Survived'})

print("\n--- Dataset Summary Statistics ---")
print(df[['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']].describe())



fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.countplot(x='Survival_Label', data=df, ax=axes[0, 0], palette=['#e74c3c', '#2ecc71'])
axes[0, 0].set_title("1. Overall Passenger Survival Count", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Status")
axes[0, 0].set_ylabel("Passenger Count")

sns.barplot(x='Sex', y='Survived', data=df, ax=axes[0, 1], ci=None, palette=['#3498db', '#e84393'])
axes[0, 1].set_title("2. Survival Rate by Gender", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Gender")
axes[0, 1].set_ylabel("Survival Rate (0 to 1)")


sns.barplot(x='Pclass', y='Survived', data=df, ax=axes[1, 0], ci=None, palette='Blues_r')
axes[1, 0].set_title("3. Survival Rate by Ticket Class (Pclass)", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Passenger Class (1 = 1st Class, 3 = 3rd Class)")
axes[1, 0].set_ylabel("Survival Rate")


sns.histplot(data=df, x='Age', hue='Survival_Label', kde=True, ax=axes[1, 1], palette=['#e74c3c', '#2ecc71'], bins=30)
axes[1, 1].set_title("4. Age Distribution vs Survival Status", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Age")
axes[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, ci=None, palette=['#3498db', '#e84393'])
plt.title("5. Survival Rate by Class and Gender", fontsize=14, fontweight='bold')
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()