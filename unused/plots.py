import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv('./MUSIC_GENRE_CLAS/music_genre.csv')

# Drop rows with missing values(where the whole row is missing)
df = df.dropna()

def replace_empty_artist_name(df):
    df['artist_name'] = df['artist_name'].replace('empty_field', 'Unknown Artist')
    return df

replace_empty_artist_name(df)

def analyze_duration(df):
    # Decide whether to use median or mean to replace -1 values in duration_ms
    # First, you should examine the distribution of the duration_ms data. You can visualize the distribution using a histogram or a box plot and also calculate the skewness of the data.
    # Let's plot the histogram of the duration_ms column excluding -1 values
    df[df['duration_ms'] > 0]['duration_ms'].hist(bins=1000)
    plt.title('Distribution of duration_ms')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate skewness
    skewness = df[df['duration_ms'] > 0]['duration_ms'].skew()
    print(f"Skewness of duration_ms: {skewness}")
    
def replace_negative_duration(df):
    # Calculate the median duration from valid entries
    median_duration = df[df['duration_ms'] > 0]['duration_ms'].median()

    # Replace -1 values with the median duration
    df['duration_ms'] = df['duration_ms'].replace(-1, median_duration)
    # Optional: Cap durations at the 95th percentile to limit the impact of extreme outliers
    #percentile_95 = df['duration_ms'].quantile(0.95)
    #df['duration_ms'] = df['duration_ms'].clip(upper=percentile_95)
    
    return df

df = replace_negative_duration(df)

def analyze_instrumentalness(df):#understood that 0s are valid values in the instrumentalness column
    # Check the distribution of other features where instrumentalness is 0
    print(df[df['instrumentalness'] == 0].describe())

    # Display summary statistics for numeric columns-for comparison
    print(df.describe())


def replace_missing_tempo(df):
    df['tempo'] = pd.to_numeric(df['tempo'].replace('?', np.nan), errors='coerce')
    mean_tempo = df['tempo'].mean()
    df['tempo'] = df['tempo'].fillna(mean_tempo)
    return df

replace_missing_tempo(df)

def analyze_tempo(df):
    df[df['tempo'] > 0]['tempo'].hist(bins=100)
    plt.title('Distribution of tempo')
    plt.xlabel('Tempo')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate skewness
    skewness = df[df['tempo'] > 0]['tempo'].skew()
    print(f"Skewness of tempo: {skewness}")



def fill_missing_tempo(df):
    #this aproach allows for more nuanced imputation of missing tempo values based on the music genre respects the typical characteristics of each genre
    if 'music_genre' in df.columns:
        for genre in df['music_genre'].unique():
            genre_mode = df[df['music_genre'] == genre]['tempo'].mode()[0]
            df.loc[(df['tempo'].isnull()) & (df['music_genre'] == genre), 'tempo'] = genre_mode

fill_missing_tempo(df)

#Feature Engineering################################
####################################################

#Creating new features from existing ones##

#interaction features

def create_interaction_features(df):
    df['energy_danceability'] = df['energy'] * df['danceability']
    
    scaler = StandardScaler()#standar scaler scales tjese features to have a mean of 0 and a standard deviation of 1
    df['loudness_scaled'] = scaler.fit_transform(df[['loudness']])
    df['loudness_energy'] = df['loudness_scaled'] * df['energy']
    
    return df

#df = create_interaction_features(df)

#aggregate features

def create_acoustic_instrumental_ratio(df):
    df['acoustic_instrumental_ratio'] = df['acousticness'] / (df['instrumentalness'] + 0.001)
    return df

#create_acoustic_instrumental_ratio(df)

#Categorical Binning of Continuous Variables

def create_categorical_features(df):
    bins = [0, 60, 90, 120, 150, 180, float('inf')]
    labels = ['very_slow', 'slow', 'moderate', 'fast', 'very_fast', 'extremely_fast']
    df['tempo_category'] = pd.cut(df['tempo'], bins=bins, labels=labels)

    df['duration_cat'] = pd.cut(df['duration_ms'], bins=[0, 180000, 240000, float('inf')], labels=['short', 'medium', 'long'])
    
    return df

#create_categorical_features(df)

#polynomial features

def generate_polynomial_features(df):

    # Initialize the PolynomialFeatures object with degree 2 (for quadratic interactions)
    poly = PolynomialFeatures(degree=2, include_bias=False)

    # Select features to transform
    features = df[['tempo', 'energy', 'danceability', 'loudness', 'acousticness']]

    # Scale features before applying polynomial transformations
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Generate polynomial features
    features_poly = poly.fit_transform(features_scaled)
    poly_feature_names = poly.get_feature_names_out(['tempo', 'energy', 'danceability', 'loudness', 'acousticness'])

    # Create a DataFrame with the new polynomial features
    df_poly = pd.DataFrame(features_poly, columns=poly_feature_names)

    # Reset indices if they do not match
    df.reset_index(drop=True, inplace=True)
    df_poly.reset_index(drop=True, inplace=True)

    # Drop original feature columns from df_poly
    original_features = ['tempo', 'energy', 'danceability', 'loudness', 'acousticness']
    df_poly.drop(original_features, axis=1, inplace=True)

    # Merge the new polynomial features back into the original DataFrame
    df = pd.concat([df, df_poly], axis=1)
    
    return df
    
    

#display the features before adding polynomial features (whether they are categorical or continuous)
#print(df.dtypes)


#df = generate_polynomial_features(df)#total number of polynomial features generated is 21

#or if it is the second time with the same column name
new_column_names = []
for col in df.columns:
    #or if it is the second time with the same column name
    if '^' in col or ' ' in col:  # Identify polynomial feature columns
        

        new_column_names.append(col + '_poly')  # Append a suffix to denote polynomial features
    else:
        new_column_names.append(col)

df.columns = new_column_names

#added polynomial features to the dataset i do not know if  they are useful or not but i will keep them for now

#########################################
#EDA- exploratory data analysis 

def plot_feature_distribution(df, feature):
    # Plot the distribution of a feature
    df[feature].hist(bins=100)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

def plot_feature_boxplot(df, feature):
    # Plot a boxplot of a feature
    sns.boxplot(data=df, x=feature)
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.show()

def plot_feature_by_genre(df, feature):
    # Plot the distribution of a feature by genre(target variable)
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='music_genre', y=feature)
    plt.title(f'{feature} by Genre')
    plt.xlabel('Genre')
    plt.ylabel(feature)
    plt.xticks(rotation=45)
    plt.show()



# Histograms for continuous features
def plot_histograms_1(df):
    for col in df.select_dtypes(include=['float64']).columns:
        plot_feature_distribution(df, col)
        #plot_feature_boxplot(df, col)
        #plot_feature_by_genre(df, col)
        
# plot_histograms(df)
# print("done")

# Boxplots for visualizing outliers
def plot_boxplots_1(df):
    for col in df.select_dtypes(include=['float64']).columns:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.show()

# Histograms for continuous features--via cptt
def plot_histograms(df):
    for col in ['popularity', 'acousticness', 'danceability', 'energy', 'loudness']:   
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True)#kde=True adds a kernel density estimate to the plot
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
#plot_histograms(df)

# Boxplots for visualizing outliers--via cptt
def plot_boxplots(df):
    for col in ['popularity', 'acousticness', 'danceability', 'energy', 'loudness', 'speechiness', 'valence', 'tempo']:
        try:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)
            plt.show()
        except Exception as e:
            print(f"Error plotting {col}: {e}")


#plot_boxplots(df)

# Check the content of each column
def check_content(df):
    for col in df.columns:
        print(f"Column: {col}")
        print(df[col].value_counts())
        print("\n")
#check_content(df)

# Scatter plots for continuous features vs. music genre--via cptt
def plot_scatterplots(df):
    for col in ['tempo', 'energy', 'danceability', 'loudness', 'acousticness']:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[col], y=df['music_genre'])
        plt.title(f'Scatter plot of {col} vs. Music Genre')
        plt.xlabel(col)
        plt.ylabel('Music Genre')
        plt.show()
#plot_scatterplots(df)

# Pairplot for selected features--via cptt
def plot_pairplot(df):
    features = ['tempo', 'energy', 'danceability', 'loudness', 'acousticness', 'music_genre']
    sns.pairplot(df[features], hue='music_genre', diag_kind='hist')
    #sns.pairplot(df[features], hue='music_genre', corner=True) # corner=True from cptt
    plt.show()

#plot_pairplot(df)

def visualize_correlation(df):

    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])  # Ensure this imports numpy as np

    # Correlation matrix
    correlation_matrix = numeric_df.corr()

    # Heatmap to visualize the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Features')
    plt.show()

    # Specific correlation with target, if target is numerically encoded; otherwise, consider ANOVA or similar tests
    # Assuming 'music_genre' is numerically encoded for this purpose
    if 'music_genre' in correlation_matrix:
        genre_corr = correlation_matrix['music_genre'].sort_values(ascending=False)
        print(genre_corr)

#visualize_correlation(df)

def visualize_simplified_correlation(df):

    # Select a meaningful subset of features
    selected_features = [
        'popularity', 'acousticness', 'danceability', 'energy', 
        'loudness', 'speechiness', 'valence', 'tempo'
    ]

    # Ensure only these selected features are used
    df_selected = df[selected_features]

    # Calculate the correlation matrix for the selected features
    correlation_matrix = df_selected.corr()

    # Heatmap to visualize the correlation matrix
    plt.figure(figsize=(10, 8))  # Adjust size as needed
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Simplified Correlation Matrix of Selected Features')
    plt.show()

# Apply the function to your DataFrame
#visualize_simplified_correlation(df)

def visualize_correlation_genre(df):
    le = LabelEncoder()
    df['music_genre_encoded'] = le.fit_transform(df['music_genre'])

    # Select a meaningful subset of features
    selected_features = [
        'popularity', 'acousticness', 'danceability', 'energy', 
        'loudness', 'speechiness', 'valence', 'tempo', 'music_genre_encoded'
    ]

    # Ensure only these selected features are used
    df_selected = df[selected_features]

    # Calculate the correlation matrix for the selected features
    correlation_matrix = df_selected.corr()

    # Heatmap to visualize the correlation matrix
    plt.figure(figsize=(12, 10))  # Adjust size as needed
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Selected Features and Music Genre')
    plt.show()

# Apply the function to your DataFrame
#visualize_correlation_genre(df)

def plot_key_frequency(df):
    # Plotting the frequency of keys within each genre
    plt.figure(figsize=(14, 8))
    sns.countplot(x='key', hue='music_genre', data=df, palette='viridis')
    plt.title('Distribution of Musical Keys Across Genres')
    plt.xlabel('Musical Key')
    plt.ylabel('Frequency')
    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.show()

#plot_key_frequency(df)

def plot_mode_frequency(df):
    # Plotting the frequency of modes within each genre
    plt.figure(figsize=(10, 6))
    sns.countplot(x='mode', hue='music_genre', data=df, palette='viridis')
    plt.title('Distribution of Mode Across Genres')
    plt.xlabel('Mode')
    plt.ylabel('Frequency')
    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

#plot_mode_frequency(df)

#Chi-Squared Test for Feature Selection -catgorical features
def encode_categorical_features(df):
    # Assuming 'key' and 'mode' are categorical features in your dataset
    encoder = LabelEncoder()
    df['key_encoded'] = encoder.fit_transform(df['key'])
    df['mode_encoded'] = encoder.fit_transform(df['mode'])
    df['music_genre_encoded'] = encoder.fit_transform(df['music_genre'])  # ensure it's encoded for this usage
    return df

encode_categorical_features(df)

# Create a contingency table and perform Chi-square test
def perform_chi_square(feature):
    contingency_table = pd.crosstab(df[feature], df['music_genre_encoded'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square test for {feature}:")
    print(f"Chi2 statistic: {chi2}, p-value: {p}\n")

# Apply the test to categorical features
#perform_chi_square('key_encoded')
#perform_chi_square('mode_encoded')


#model-based feature selection-random forest

# Scatter plots for continuous features vs. music genre--via cptt
def plot_scatterplots_test(df):
    #for all numerical features
    for col in df.select_dtypes(include=['float64']).columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[col], y=df['music_genre'])
        plt.title(f'Scatter plot of {col} vs. Music Genre')
        plt.xlabel(col)
        plt.ylabel('Music Genre')
        plt.show()
#plot_scatterplots_test(df)

# Boxplots for visualizing outliers--via cptt
def plot_boxplots_test(df):
    for col in df.select_dtypes(include=['float64']).columns:
        try:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)
            plt.show()
        except Exception as e:
            print(f"Error plotting {col}: {e}")

#plot_boxplots_test(df)

# Histograms for continuous features--via cptt
def plot_histograms_test(df):
    #for all numerical features
    for col in df.select_dtypes(include=['float64']).columns: 
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True)#kde=True adds a kernel density estimate to the plot
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
plot_histograms_test(df)
