import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split



# Load the dataset
df = pd.read_csv('./MUSIC_GENRE_CLAS/music_genre.csv')


def explore_dataset(df):
    # Display the first few rows of the DataFrame
    #print(df.head())

    # Display info about data types and number of non-null values
    print(df.info()) # some of the tempo values seem to be strings it is because missing values in this column are represented as '?' in the dataset

    # Display summary statistics for numeric columns
    print(df.describe())

    # Optionally, explore the number of unique values in categorical columns
    for col in df.select_dtypes(include=['object']).columns: 
        print(f"{col} has {df[col].nunique()} unique values")

    # Check for missing values
    print(df.isnull().sum()) # 5 rows are missing


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

#Feature Selection################################
####################################################
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

#new best parameters found with the grid search Best parameters found:  {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}
#model-based feature selection-random forest
def feature_ranking(df):
    # Dropping non-useful non-numeric columns
    X = df.drop(['music_genre_encoded', 'artist_name', 'track_name', 'obtained_date', 'key', 'mode', 'music_genre','instance_id'], axis=1)

    # Assuming 'tempo_category' and 'duration_cat' need to be encoded if they haven't been already
    if 'tempo_category' in X.columns:
        X['tempo_category_encoded'] = LabelEncoder().fit_transform(X['tempo_category'])
        X.drop('tempo_category', axis=1, inplace=True)
    if 'duration_cat' in X.columns:
        X['duration_cat_encoded'] = LabelEncoder().fit_transform(X['duration_cat'])
        X.drop('duration_cat', axis=1, inplace=True)

    y = df['music_genre_encoded']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the random forest
    forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42
    )
    forest.fit(X_train, y_train)

    # Predict on the test set
    y_pred = forest.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Get feature importances
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature rankings
    # print("Feature ranking:")
    # for f in range(X.shape[1]):
    #     print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]})")
    
    return accuracy # Return the accuracy for comparison

#feature_ranking(df)
#dropped columns are: 'music_genre_encoded', 'artist_name', 'track_name', 'obtained_date', 'key', 'mode', 'music_genre'




def train_random_forest(df):
    important_features = [
        'popularity','loudness','instrumentalness','speechiness',
        'acousticness','danceability','energy','valence',
        'duration_ms','tempo'
  


        # Add other features based on your importance threshold
    ]
    #tried with other combinations too but this one gave the best accuracy

    

    X = df[important_features]
    y = df['music_genre_encoded']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the random forest on the refined feature set
    #initialize the random forest classifier with the hyperparameters that were found to be optimal
    forest_refined = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42
    )
    forest_refined.fit(X_train, y_train)

    # Predict on the test set
    y_pred = forest_refined.predict(X_test)

    # Calculate accuracy
    accuracy_refined = accuracy_score(y_test, y_pred)
    print(f"Accuracy with refined features: {accuracy_refined:.2f}")

    # Compare this accuracy with the previous model's accuracy
    return accuracy_refined

#train_random_forest(df)
#improvement was not significant from 0.53 to 0.54

def train_random_forest_with_hyperparameter_tuning(df):
    # Assuming the important features and encoded target are already defined
    X = df[['popularity','loudness','instrumentalness','speechiness',
        'acousticness','danceability','energy','valence',
        'duration_ms','tempo']]
    y = df['music_genre_encoded']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set the parameters by cross-validation
    param_grid = {
        'n_estimators': [100, 200, 300,400],  # Number of trees in random forest
        'max_features': [ 'sqrt'],  # Number of features to consider at every split - 'auto' gave nothing additional
        'max_depth': [10, 20, 30, None],  # Maximum number of levels in tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at each leaf node
    }

    # Initialize the classifier
    rf = RandomForestClassifier(random_state=42)

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Best parameters found
    print("Best parameters found: ", grid_search.best_params_)

    # Evaluate the best grid search model on the test set
    best_grid = grid_search.best_estimator_
    y_pred = best_grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with the best parameters: {accuracy:.2f}")

#train_random_forest_with_hyperparameter_tuning(df)
#Best parameters found:  {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
#Accuracy: 0.53
#Accuracy with refined features: 0.56 - after hyperparameter tuning for the random forest classifier




#PCA (Principal Component Analysis): 
def apply_pca(df):
    ## Selecting the core features based on their importance
    core_features = df[['popularity', 'speechiness', 'loudness', 'instrumentalness', 
                        'danceability', 'energy', 'acousticness', 'valence', 'duration_ms', 
                        'tempo', 'mode_encoded', 'liveness', 'key_encoded']]

    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(core_features)

    # Applying PCA
    pca = PCA(n_components=0.95)  # Retain 95% of the variance
    X_pca = pca.fit_transform(X_scaled)
    print("Number of components:", pca.n_components_)

    # Visualizing the variance explained by each component
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

    # Optionally, plot the first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['music_genre_encoded'], cmap='viridis', edgecolor='k', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.title('PCA: Projection onto the first two principal components')
    plt.show()

    


#apply_pca(df)

#PCA suggest that the data is not separable in the first two principal components
#the PCA also suggests that the total of 10 components are needed to explain 95% of the variance in the data which also gets along with the features from the random forest classifier
#the 10 components in Random Forest Classifier also give the most accurate results


#####DROPPPP#####
#print(df.info())
#drop key and mode columns,instance_id,artist_name,track_name,obtained_date
#df = df.drop(['key', 'mode', 'instance_id', 'artist_name', 'track_name', 'obtained_date'], axis=1)






#t-SNE (t-distributed Stochastic Neighbor Embedding)

def apply_tsne0(df):
    tsne = TSNE(n_components=2, random_state=42)
    X=df[[
        'popularity', 'speechiness', 'loudness', 'instrumentalness', 
        'danceability', 'energy', 'acousticness', 'valence', 'duration_ms', 
        'tempo', 'mode_encoded', 'liveness', 'key_encoded']]
    y = df['music_genre_encoded']
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title('t-SNE visualization of Dataset')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.colorbar()
    plt.show()

#apply_tsne0(df)

def apply_tsne(df):
    """
    Applies t-SNE dimensionality reduction to visualize high-dimensional data.
    Args:
    df (DataFrame): A pandas DataFrame with the required features and encoded labels.

    Returns:
    None: This function plots the t-SNE visualization.
    """
    # Check if required columns are present
    required_columns = ['popularity', 'speechiness', 'loudness', 'instrumentalness', 
                        'danceability', 'energy', 'acousticness', 'valence', 'duration_ms', 
                        'tempo', 'mode_encoded', 'liveness', 'key_encoded', 'music_genre_encoded']
    if not all(column in df.columns for column in required_columns):
        raise ValueError("Dataframe does not contain all required columns.")

    # Select features and target
    X = df[required_columns[:-1]]
    y = df['music_genre_encoded']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title('t-SNE visualization of Dataset')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.colorbar(scatter)
    plt.show()

# Usage
#apply_tsne(df)  # Replace 'dataframe' with your actual dataframe variable name

def test_tsne(features, labels, perplexities, learning_rates, iterations=1000):
    results = []
    for perplexity in perplexities:
        for learning_rate in learning_rates:
            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, 
                        n_iter=iterations, random_state=42)
            X_tsne = tsne.fit_transform(features)
            # Visualize or save the result
            plt.figure(figsize=(10, 8))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
            plt.title(f't-SNE with Perplexity {perplexity} and Learning Rate {learning_rate}')
            plt.xlabel('t-SNE Feature 1')
            plt.ylabel('t-SNE Feature 2')
            plt.colorbar()
            plt.show()
            # Log the settings and any metrics or observations
            results.append((perplexity, learning_rate, X_tsne))
    return results

# # Define the parameters to test
# perplexities = [5, 30, 50]
# learning_rates = [100, 200,500]
# X=df[[
#     'popularity', 'speechiness', 'loudness', 'instrumentalness', 
#     'danceability', 'energy', 'acousticness', 'valence', 'duration_ms', 
#     'tempo', 'mode_encoded', 'liveness', 'key_encoded']]
# y = df['music_genre_encoded']
# test_tsne(X, y, perplexities, learning_rates)


def apply_pca_re(df):
    # Selecting the core features based on their importance
    core_features = df[['popularity', 'speechiness', 'loudness', 'instrumentalness', 
                        'danceability', 'energy', 'acousticness', 'valence', 'duration_ms', 
                        'tempo', 'mode_encoded', 'liveness', 'key_encoded']]

    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(core_features)

    # Applying PCA
    pca = PCA(n_components=0.95)  # Retain 95% of the variance
    X_pca = pca.fit_transform(X_scaled)

    # Output the number of components
    print("Number of components:", pca.n_components_)

    # Extract and display component loadings
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                            index=core_features.columns)
    print(loadings)

    # Visualizing the variance explained by each component
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

    # Optionally, plot the first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['music_genre_encoded'], cmap='viridis', edgecolor='k', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.title('PCA: Projection onto the first two principal components')
    plt.show()

# Replace 'df' with your actual DataFrame name when calling this function
#apply_pca_re(df)

def plot_correlation_matrix(df):
    # Compute the correlation matrix for the features in your dataset
    correlation_matrix = df[['popularity', 'speechiness', 'loudness', 'instrumentalness', 
                             'danceability', 'energy', 'acousticness', 'valence', 'duration_ms', 
                             'tempo', 'mode_encoded', 'liveness', 'key_encoded']].corr()

    # Plot the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Features')
    plt.show()

# Replace 'df' with your actual DataFrame name when calling this function
#plot_correlation_matrix(df)

def calculate_reconstruction_error(df):
        # Select the core features based on their importance
    core_features = df[['popularity', 'speechiness', 'loudness', 'instrumentalness', 
                        'danceability', 'energy', 'acousticness', 'valence', 'duration_ms', 
                        'tempo', 'mode_encoded', 'liveness', 'key_encoded']]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(core_features)

    # Apply PCA to retain 95% of the variance
    pca = PCA(n_components=0.95)
    pca.fit(X_scaled)

    # Reconstruct data from the PCA components
    X_reconstructed = pca.inverse_transform(pca.transform(X_scaled))

    # Calculate and return the mean squared error of reconstruction
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    return mse

# You can calculate the reconstruction error after applying PCA to see the impact of dimensionality reduction
# Replace 'pca' and 'X_scaled' with your actual PCA model and scaled data


#error = calculate_reconstruction_error(df)
#print("Reconstruction MSE:", error)

def train_and_evaluate_decision_tree(X, y):
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Decision Tree
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred_tree = tree_model.predict(X_test)
    print(classification_report(y_test, y_pred_tree))
    print(confusion_matrix(y_test, y_pred_tree))