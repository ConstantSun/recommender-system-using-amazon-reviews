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

logging.basicConfig(filename="log/result.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        logging.info(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# Load the Dataset and Add headers
def load_dataset(filename: str = "ratings_Electronics (1).csv"):
    # filename: tên file data 
    # return : pandas frame data 
    data = pd.read_csv(filename, names=['userId', 'productId','Rating','timestamp'])
    return data


def data_info_1(data):
    # Display the data
    logging.info(f"Display the data\n{electronics_data.head()}")
    #Shape of the data
    logging.info(f"Shape of the data\n{electronics_data.shape}")


def taking_subset_dataset(data):
    #Taking subset of the dataset
    data = data.iloc[:1048576,0:]
    return data

def data_info_2(data):
    #Check the datatypes
    logging.info(f"Check the datatypes\n{data.dtypes}")
    logging.info(data.info())
    #Five point summary 
    logging.info(f"Five point summary\n{data.describe()['Rating'].T}")
    #Find the minimum and maximum ratings
    logging.info(f"Minimum rating is: {data.Rating.min()}")
    logging.info(f"Maximum rating is: {data.Rating.max()}")
    # Handling Missing values
    # Check for missing values
    logging.info(f"Number of missing values across columns: \n{data.isnull().sum()}")


def visualize_rating_distribution(data):
    # Check the distribution of the rating
    with sns.axes_style('white'):
        g = sns.factorplot("Rating", data=data, aspect=2.0,kind='count')
        g.set_ylabels("Total number of ratings")
    plt.savefig('log/rating_dis.png')


def data_info_3(data):
    # Unique Users and products
    logging.info("Total data ")
    logging.info("-"*50)
    logging.info(f"\nTotal no of ratings : {data.shape[0]}")
    logging.info(f"Total No of Users   : {len(np.unique(data.userId))}")
    logging.info(f"Total No of products  : {len(np.unique(data.productId))}")


def drop_time_col(data):
    #Dropping the Timestamp column
    data.drop(['timestamp'], axis=1,inplace=True)
    return data


def analyzing_rating(data):
    # Analyzing the rating
    # Analysis of rating given by the user 
    no_of_rated_products_per_user = data.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
    logging.info("Analysis of rating given by the user")
    logging.info(no_of_rated_products_per_user.head())
    logging.info(no_of_rated_products_per_user.describe())

def visualize_quantiles_values(data):
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
    new_df=data.groupby("productId").filter(lambda x:x['Rating'].count() >=50)
    return new_df


# def get_popularity_based_recommendation(data):
#     # data: pandas dataframe
#     # Popularity Based Recommendation
#     # Getting the new dataframe which contains users who has given 50 or more ratings
#     new_df=data.groupby("productId").filter(lambda x:x['Rating'].count() >=50)
    
#     return no_of_ratings_per_product


def visualize_num_ratings_per_product(new_df):
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
    #Average rating of the product 
    logging.info(f"Average rating of the product:\n{data.groupby('productId')['Rating'].mean().head()}")
    logging.info(data.groupby('productId')['Rating'].mean().sort_values(ascending=False).head())

    #Total no of rating for product
    logging.info(f"Total no of rating for product:\n{data.groupby('productId')['Rating'].count().sort_values(ascending=False).head()}")


def get_ratings_mean_count(data):
    ratings_mean_count = pd.DataFrame(new_df.groupby('productId')['Rating'].mean())
    ratings_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('productId')['Rating'].count())
    return ratings_mean_count

def data_info_5(data):
    logging.info(data.head())
    logging.info(data['rating_counts'].max())

def visualize_ratings_mean_count(ratings_mean_count):
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
    # data: pandas dataframe
    popular_products = pd.DataFrame(data.groupby('productId')['Rating'].count())
    return popular_products


def visualize_most_popular_products(popular_products):
    most_popular = popular_products.sort_values('Rating', ascending=False)
    most_popular.head(30).plot(kind = "bar")
    plt.savefig("log/most_popular.png")


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
