import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import zipfile
import io
import requests

class MovieLensDataset(Dataset):  

    def __init__(self, dataset_version="small", data_dir="data"):

        print("Initializing MovieLensDataset...")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if dataset_version == "small":
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            local_zip_path = os.path.join(data_dir,"ml-latest-small.zip")
            file_path = 'ml-latest-small/ratings.csv'
            parquet_path = os.path.join(data_dir,"ml-latest-small.parquet")
        elif dataset_version == "full":
            url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
            local_zip_path = os.path.join(data_dir,"ml-latest.zip")
            file_path = 'ml-latest/ratings.csv'
            parquet_path = os.path.join(data_dir, "ml-latest.parquet")
        else:
            raise ValueError("Invalid dataset_version. Choose 'small' or 'full'.")

        if os.path.exists(parquet_path):
            print(f"Loading dataset from {parquet_path}...")
            movielens = pd.read_parquet(parquet_path)
        else:
            if not os.path.exists(local_zip_path):
                print(f"Downloading {dataset_version} dataset from {url}...")
                response = requests.get(url)
                with open(local_zip_path, "wb") as f:
                    f.write(response.content)

            with zipfile.ZipFile(local_zip_path,"r") as z:
                with z.open(file_path) as f:
                    movielens = pd.read_csv(f, usecols=['userId', 'movieId', 'rating'], low_memory=True)
                movielens.to_parquet(parquet_path, index=False)

        movielens['liked'] = (movielens['rating'] >= 4).astype(int)
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        movielens['user'] = self.user_encoder.fit_transform(movielens['userId'])
        movielens['movie'] = self.movie_encoder.fit_transform(movielens['movieId'])
        self.train_df, self.test_df = train_test_split(movielens, test_size=0.2,random_state=42)

    def get_data(self, split="train"):

        if split == "train":
            data = self.train_df
        elif split == "test":
            data = self.test_df
        else:
            raise ValueError("Invalid split. Choose 'train' or 'test'.")
        dense_features = torch.tensor(data[['user', 'movie']].values, dtype=torch.long)
        labels = torch.tensor(data[['rating', 'liked']].values,dtype=torch.float32)
        return dense_features, labels

    def get_encoders(self):
        return self.user_encoder, self.movie_encoder

# Example usage with a single dataset object
print("Creating MovieLens dataset...")
# Feel free to use dataset_version="full" if you are using a GPU
dataset = MovieLensDataset(dataset_version="small")
print("Getting training data...")
train_dense_features, train_labels = dataset.get_data(split="train")
print("Getting testing data...")
test_dense_features, test_labels = dataset.get_data(split="test")
# Create DataLoader for training and testing
train_loader = DataLoader(torch.utils.data.TensorDataset(train_dense_features, train_labels), batch_size=64, shuffle=True)
test_loader = DataLoader(torch.utils.data.TensorDataset(test_dense_features, test_labels), batch_size=64, shuffle=False)
print("Accessing encoders...")
user_encoder, movie_encoder = dataset.get_encoders()
print("Setup complete.")