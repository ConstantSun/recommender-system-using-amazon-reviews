# Import library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import logging


from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(filename="log/result.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        logging.info(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


def load_dataset(filename: str = "ratings_Electronics (1).csv"):
    """
    This function for: Load the Dataset and Add headers
    Args:
        filename: tên file data 
    Return:    
        pandas frame data 
    """
    data = pd.read_csv(filename, names=['userId', 'productId','Rating','timestamp'])
    return data


def data_info_1(data):
    """
    This function for: 
        Display the data
        Shape of the data
    Args:
        data
    Return:    
        
    """    
    logging.info(f"Display the data\n{electronics_data.head()}")
    logging.info(f"Shape of the data\n{electronics_data.shape}")


def taking_subset_dataset(data):
    """
    This function for: Taking subset of the dataset
    Args:
        data
    Return:    
        data
    """       
    data = data.iloc[:1048576,0:]
    return data

def data_info_2(data):
    """
    This function for: 
        Check the datatypes
        Five point summary 
        Find the minimum and maximum ratings
        Check for missing values
    Args:
        data
    Return:    
        
    """       
    logging.info(f"Check the datatypes\n{data.dtypes}")
    logging.info(data.info())
    
    logging.info(f"Five point summary\n{data.describe()['Rating'].T}")
    logging.info(f"Minimum rating is: {data.Rating.min()}")
    logging.info(f"Maximum rating is: {data.Rating.max()}")
    logging.info(f"Number of missing values across columns: \n{data.isnull().sum()}")


def visualize_rating_distribution(data):
    """
    This function for: 
    Args:
        Check the distribution of the rating
        
    Return:    
        
    """       
    with sns.axes_style('white'):
        g = sns.factorplot("Rating", data=data, aspect=2.0,kind='count')
        g.set_ylabels("Total number of ratings")
    plt.savefig('log/rating_dis.png')


def data_info_3(data):
    """
    This function for: 
        Print Unique Users and products
    Args:
        data
    Return:    
        
    """       
    logging.info("Total data ")
    logging.info("-"*50)
    logging.info(f"\nTotal no of ratings : {data.shape[0]}")
    logging.info(f"Total No of Users   : {len(np.unique(data.userId))}")
    logging.info(f"Total No of products  : {len(np.unique(data.productId))}")


def drop_time_col(data):
    """
    This function for: Dropping the Timestamp column
    Args:
        data
    Return:    
        
    """       
    data.drop(['timestamp'], axis=1,inplace=True)
    return data


def analyzing_rating(data):
    """
    This function for: 
        Analyzing the rating
        Analysis of rating given by the user     
    Args:
        data
    Return:    
        
    """       
    no_of_rated_products_per_user = data.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
    logging.info("Analysis of rating given by the user")
    logging.info(no_of_rated_products_per_user.head())
    logging.info(no_of_rated_products_per_user.describe())


def visualize_quantiles_values(data):
    """
    This function for: visualize quantiles values
    Args:
        data
    Return:    
        
    """       
    no_of_rated_products_per_user = data.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
    quantiles = no_of_rated_products_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')
    plt.figure(figsize=(10,10))
    plt.title("Quantiles and their Values")
    quantiles.plot()
    # quantiles with 0.05 difference
    plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")
    # quantiles with 0.25 difference
    plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
    plt.ylabel('No of ratings by user')
    plt.xlabel('Value at the quantile')
    plt.legend(loc='best')
    plt.savefig("log/quantiles_their_vals.png")
    plt.show()
    logging.info('\n No of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 50)) )


def get_dataframe_which_users_more_50_ratings(data):
    """
    This function for: get dataframe which users more 50 ratings
    Args:
        data
    Return:    
        
    """       
    new_df=data.groupby("productId").filter(lambda x:x['Rating'].count() >=50)
    return new_df


def visualize_num_ratings_per_product(new_df):
    """
    This function for: visualize number of ratings per product
    Args:
        new_df: pandas dataframe
    Return:    
        
    """       
    no_of_ratings_per_product = new_df.groupby(by='productId')['Rating'].count().sort_values(ascending=False)
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = plt.gca()
    plt.plot(no_of_ratings_per_product.values)
    plt.title('# RATINGS per Product')
    plt.xlabel('Product')
    plt.ylabel('No of ratings per product')
    ax.set_xticklabels([])
    plt.savefig("log/num_ratings_per_product.png")
    plt.show()


def data_info_4(data):
    """
    This function for: printing:
        Average rating of the product
        Total no of rating for product
    Args:
        data
    Return:            
    """           
    logging.info(f"Average rating of the product:\n{data.groupby('productId')['Rating'].mean().head()}")
    logging.info(data.groupby('productId')['Rating'].mean().sort_values(ascending=False).head())
    logging.info(f"Total no of rating for product:\n{data.groupby('productId')['Rating'].count().sort_values(ascending=False).head()}")


def get_ratings_mean_count(data):
    """
    This function for: get_ratings_mean_count
    Args:
        data
    Return:    
        
    """       
    ratings_mean_count = pd.DataFrame(new_df.groupby('productId')['Rating'].mean())
    ratings_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('productId')['Rating'].count())
    return ratings_mean_count

def data_info_5(data):
    """
    This function for: printing data information
    Args:data
    Return:    
        
    """       
    logging.info(data.head())
    logging.info(data['rating_counts'].max())


def visualize_ratings_mean_count(ratings_mean_count):
    """
    This function for: visualize ratings mean count
    Args:
        ratings_mean_count
    Return:    
        
    """       
    plt.figure(figsize=(8,6))
    plt.rcParams['patch.force_edgecolor'] = True
    ratings_mean_count['rating_counts'].hist(bins=50)
    plt.savefig("log/ratings_counts.png")

    plt.figure(figsize=(8,6))
    plt.rcParams['patch.force_edgecolor'] = True
    ratings_mean_count['Rating'].hist(bins=50)
    plt.savefig("log/rating.png")

    plt.figure(figsize=(8,6))
    plt.rcParams['patch.force_edgecolor'] = True
    sns.jointplot(x='Rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)
    plt.savefig("log/rating_rating_counts.png")


def get_popular_products(data):
    """
    This function for: 
        get_popular_products
    Args:
        data: pandas dataframe
    Return:    
        
    """       
    
    popular_products = pd.DataFrame(data.groupby('productId')['Rating'].count())
    return popular_products


def visualize_most_popular_products(popular_products):
    """
    This function for: 
    Args:
        
    Return:    
        
    """       
    most_popular = popular_products.sort_values('Rating', ascending=False)
    most_popular.head(30).plot(kind = "bar")
    plt.savefig("log/most_popular.png")



########### 
# Collaberative filtering (Item-Item recommedation)

def read_data(data):
    """
    This function for: Reading the dataset
    Args: 
        data        
    Return:    
        
    """       
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(new_df,reader)
    return data


def separate_data(data):
    """
    This function for: Splitting the dataset
    Args: 
        data        
    Return:    
        
    """  
    trainset, testset = train_test_split(data, test_size=0.3,random_state=10)
    return trainset, testset 


def make_alg_and_test(trainset, testset):
    """
    This function for: create the algorithm and run the algorithm on test dataset.
    Args: 
        trainset, testset        
    Return:    
    """

    # Use user_based true/false to switch between user-based or item-based collaborative filtering
    algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
    algo.fit(trainset)

    # run the trained model against the testset
    test_pred = algo.test(testset)

    logging.info(test_pred)

    # get RMSE
    logging.info("Item-based Model : Test Set")
    logging.info(accuracy.rmse(test_pred, verbose=True))

####
# Model-based collaborative filtering system

def get_ratings_matrix(new_df):
    """
    This function for: create the algorithm and run the algorithm on test dataset.
    Args: 
        trainset, testset        
    Return:    
    """

    new_df1=new_df.head(10000)
    ratings_matrix = new_df1.pivot_table(values='Rating', index='userId', columns='productId', fill_value=0)
    logging.info(f"Rating matrix head:\n{ratings_matrix.head()}")
    logging.info(f"Rating matrix shape:\n{ratings_matrix.shape}")
    return ratings_matrix


def transpose_matrix(ratings_matrix):
    """
    This function for: traspose the matrix 
    Args:
        ratings_matrix
    Return:    
        
    """       
    X = ratings_matrix.T
    logging.info(X.head())
    logging.info(X.shape)
    return X



def decompose_matrix(X):
    """
    This function for: Decomposing the Matrix
    Args:
        X
    Return:            
    """      
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    logging.info(decomposed_matrix.shape)
    return decomposed_matrix

def get_correlation_matrix(decomposed_matrix):
    """
    This function for: 
    Args:
        X
    Return:            
    """     
    #Correlation Matrix
    correlation_matrix = np.corrcoef(decomposed_matrix)
    logging.info(correlation_matrix.shape)
    return correlation_matrix

def get_product_id(X, i):
    """
    This function for: 
    Args:
        X
    Return:            
    """       
    # Index # of product ID purchased by customer
    product_names = list(X.index)
    product_ID = product_names.index(i)
    logging.info(product_ID)
    return product_ID


def show_top25_highly_correlated_products(X):
    """
    This function for: 
        Recommending top 25 highly correlated products in sequence
    Args:
        X
    Return:            
    """        
    Recommend = list(X.index[correlation_product_ID > 0.65])
    # Removes the item already bought by the customer
    Recommend.remove(i) 
    logging.info(Recommend[0:24])

if __name__ == "__main__":

    electronics_data = load_dataset()
    data_info_1(electronics_data)
    electronics_data = taking_subset_dataset(electronics_data)
    data_info_2(electronics_data)
    visualize_rating_distribution(electronics_data)
    data_info_3(electronics_data)

    electronics_data = drop_time_col(electronics_data)
    analyzing_rating(electronics_data)

    visualize_quantiles_values(electronics_data)

    new_df = get_dataframe_which_users_more_50_ratings(electronics_data)

    visualize_num_ratings_per_product(new_df)

    data_info_4(new_df)

    ratings_mean_count = get_ratings_mean_count(new_df)

    data_info_5(ratings_mean_count)

    visualize_ratings_mean_count(ratings_mean_count)

    popular_products = get_popular_products(new_df)

    visualize_most_popular_products(popular_products)

    data = read_data(new_df)

    trainset, testset = separate_data(data)
    make_alg_and_test(trainset, testset)
    ratings_matrix = get_ratings_matrix(new_df)

    X = transpose_matrix(ratings_matrix)    

    decomposed_matrix = decompose_matrix(X)

    correlation_matrix = get_correlation_matrix(decomposed_matrix)

    i = "B00000K135"
    product_ID = get_product_id(X, i)

    # Correlation for all items with the item purchased by this customer based on 
    # items rated by other customers people who bought the same product
    correlation_product_ID = correlation_matrix[product_ID]
    show_top25_highly_correlated_products(X)


