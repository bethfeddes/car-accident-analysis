# Install necessary packages
import pandas as pd 
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load data set
df = pd.read_csv("C:/Users/befed/Downloads/US_Accidents_March23_sampled_500k.csv")

# Drop numeric columns not needed for Apriori
df_apriori = df.drop(columns=['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Wind_Chill(F)'])

# Fill empty numeric fields with median
numeric_cols_small_missing = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)']
df_apriori[numeric_cols_small_missing] = df_apriori[numeric_cols_small_missing].fillna(df_apriori[numeric_cols_small_missing].median())

# Fill missing categorical data with 'Unknown'
categorical_cols = df_apriori.select_dtypes(include='object').columns
df_apriori[categorical_cols] = df_apriori[categorical_cols].fillna('Unknown')

# Bin important numeric categories
df_apriori['Precipitation_cat'] = pd.cut(
    df_apriori['Precipitation(in)'],
    bins=[-0.01, 0, 0.1, 0.5, 2, 10],
    labels=['None', 'Light', 'Moderate', 'Heavy', 'Extreme']
)

df_apriori['Wind_Speed_cat'] = pd.cut(
    df_apriori['Wind_Speed(mph)'],
    bins=[-0.01, 5, 15, 30, 50, 100],
    labels=['Calm', 'Breeze', 'Windy', 'Strong', 'Extreme']
)

df_apriori['Visibility_cat'] = pd.cut(
    df_apriori['Visibility(mi)'],
    bins=[-0.01, 1, 3, 6, 10, 100],
    labels=['Very Low', 'Low', 'Moderate', 'Good', 'Excellent']
)

# Replace missing binned values with 'Unknown'
df_apriori['Precipitation_cat'] = df_apriori['Precipitation_cat'].cat.add_categories('Unknown')
df_apriori['Precipitation_cat'] = df_apriori['Precipitation_cat'].fillna('Unknown')

df_apriori['Wind_Speed_cat'] = df_apriori['Wind_Speed_cat'].cat.add_categories('Unknown')
df_apriori['Wind_Speed_cat'] = df_apriori['Wind_Speed_cat'].fillna('Unknown')

df_apriori['Visibility_cat'] = df_apriori['Visibility_cat'].cat.add_categories('Unknown')
df_apriori['Visibility_cat'] = df_apriori['Visibility_cat'].fillna('Unknown')

# Drop original numeric columns
df_apriori = df_apriori.drop(columns=['Precipitation(in)', 'Wind_Speed(mph)','Visibility(mi)'])

# Filter for Illinois
df_il = df_apriori[df_apriori['State']=='IL']

# Choose columms to use in algorithm
cols_to_use = [
    'Severity', 'Weather_Condition','Precipitation_cat','Wind_Speed_cat','Visibility_cat','Bump','Crossing',
    'Roundabout','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset'
]

df_il_subset = df_il[cols_to_use].astype(str)


# Convert to transactions
transactions = []

for _, row in df_il_subset.iterrows():
    items = []
    
    for col in df_il_subset.columns:
        val = row[col]
        
        # Only include TRUE for boolean columns
        if val == True or val == 'True':
            items.append(col)
        
        # For non-boolean columns, include normally
        elif val not in ['False', False]:
            items.append(f"{col}={val}")
    
    transactions.append(items)
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_array, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = fpgrowth(df_onehot, min_support=0.003 , use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.4)

''' SEVERITY 2 RULES'''

severity_2_rules = rules[
    rules['consequents'].apply(lambda x: 'Severity=2' in x)
]

# Keep only rules where Severity=2 is the ONLY consequent
severity_2_rules = severity_2_rules[
    severity_2_rules['consequents'] == frozenset({'Severity=2'})
]

# Limit rule size (makes output cleaner)
severity_2_rules = severity_2_rules[
    severity_2_rules['antecedents'].apply(lambda x: len(x) <= 3)
]

# Filter for meaningful lift
severity_2_rules = severity_2_rules[severity_2_rules['lift'] > 1.5]

# Remove duplicates
severity_2_rules = severity_2_rules.drop_duplicates(subset=['antecedents','consequents'])

# Sort by strength
severity_2_rules = severity_2_rules.sort_values(by='lift', ascending=False)

print("\n=== TOP RULES FOR SEVERITY 2 ===\n")

for _, row in severity_2_rules.head(10).iterrows():
    antecedent = ', '.join(list(row['antecedents']))
    print(f"IF {antecedent} THEN Severity=2")
    print(f"Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f}\n")

''' SEVERITY 3-4 RULES '''

severity_3_4_rules = rules[
    rules['consequents'].apply(lambda x: 'Severity=3' in x or 'Severity=4' in x)
]

# Keep ONLY severity in consequent
severity_3_4_rules = severity_3_4_rules[
    severity_3_4_rules['consequents'].apply(lambda x: x <= {'Severity=3','Severity=4'})
]

# Limit rule size (makes output cleaner)
severity_3_4_rules = severity_3_4_rules[
    severity_3_4_rules['antecedents'].apply(lambda x: len(x) <= 3)
]

# Filter strong rules
severity_3_4_rules = severity_3_4_rules[severity_3_4_rules['lift'] > 1.5]

# Remove duplicates
severity_3_4_rules = severity_3_4_rules.drop_duplicates(subset=['antecedents','consequents'])

# Sort
severity_3_4_rules = severity_3_4_rules.sort_values(by='lift', ascending=False)

print("\n=== TOP RULES FOR SEVERITY 3 & 4 ===\n")

for _, row in severity_3_4_rules.head(15).iterrows():
    antecedent = ', '.join(list(row['antecedents']))
    consequent = ', '.join(list(row['consequents']))
    print(f"IF {antecedent} THEN {consequent}")
    print(f"Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f}\n")
