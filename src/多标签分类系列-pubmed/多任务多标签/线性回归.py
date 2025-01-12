import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import zipfile
import ray
from ray import serve
import torch
from pyarrow.dataset import dataset
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import requests
ray.init()

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


class MultiTaskMovieLensModel(nn.Module):
    def __init__(self, n_users, n_movies, embedding_size, hidden_size):
        super(MultiTaskMovieLensModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.movie_embedding = nn.Embedding(n_movies, embedding_size)
        self.shared_layer = nn.Linear(embedding_size*2, hidden_size)
        self.shared_activation = nn.ReLU()
        self.task1_fc = nn.Linear(hidden_size, 1)
        self.task2_fc = nn.Linear(hidden_size, 1)
        self.task2_activation = nn.Sigmoid()

    def forward(self, x):
        user = x[:, 0]
        movie = x[:, 1]
        user_embed = self.user_embedding(user)
        movie_embed = self.movie_embedding(movie)
        combined = torch.cat((user_embed, movie_embed), dim=1)
        shared_out = self.shared_activation(self.shared_layer(combined))
        rating_out = self.task1_fc(shared_out)
        linked_out = self.task2_fc(shared_out)
        liked_out = self.task2_activation(linked_out)

        return rating_out, liked_out


# ==============================准备数据============================================
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


# =======================模型构建与训练======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
embedding_size = 16
hidden_size = 32
n_users = len(dataset.get_encoders()[0].classes_)
n_movies = len(dataset.get_encoders()[1].classes_)
model = MultiTaskMovieLensModel(n_users, n_movies, embedding_size, hidden_size).to(device)
criterion_rating = nn.MSELoss()
criterion_liked = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_rating_loss = []
train_liked_loss = []
eval_rating_loss = []
eval_liked_loss = []
epochs = 10

initial_loss_rating = None
initial_loss_liked = None

for epoch in range(epochs):
    model.train()
    run_loss_rating = 0.0
    run_loss_liked = 0.0
    for dense_features, labels in train_loader:
        optimizer.zero_grad()

        dense_features = dense_features.to(device)
        labels = labels.to(device)

        rating_pred, liked_pred = model(dense_features)
        rating_target = labels[:, 0].unsqueeze(1)
        liked_target = labels[:, 1].unsqueeze(1)

        loss_rating = criterion_rating(rating_pred, rating_target)
        loss_liked = criterion_liked(liked_pred, liked_target)

        if initial_loss_rating is None:
            initial_loss_rating = loss_rating.item()
        if initial_loss_liked is None:
            initial_loss_liked = loss_liked.item()

        loss = (loss_rating/initial_loss_rating) + (loss_liked/initial_loss_liked)

        loss.backward()
        optimizer.step()

        run_loss_rating += loss_rating.item()
        run_loss_liked += loss_liked.item()

    train_rating_loss.append(run_loss_rating / len(train_loader))
    train_liked_loss.append(run_loss_liked / len(train_loader))

    model.eval()
    run_loss_rating = 0.0
    run_loss_liked = 0.0

    with torch.no_grad():
       for dense_features, labels in test_loader:
           dense_features = dense_features.to(device)
           labels = labels.to(device)

           rating_pred, liked_pred = model(dense_features)
           rating_target = labels[:, 0].unsqueeze(1)
           liked_target = labels[:, 1].unsqueeze(1)

           loss_rating = criterion_rating(rating_pred, rating_target)
           loss_liked = criterion_liked(liked_pred, liked_target)

           run_loss_rating += loss_rating.item()
           run_loss_liked += loss_liked.item()

    eval_rating_loss.append(run_loss_rating / len(test_loader))
    eval_liked_loss.append(run_loss_liked / len(test_loader))

    print(f"Epoch {epoch+1}/{epochs}, Train Loss Rating: {train_rating_loss[-1]:.4f}, Train Loss Liked: {train_liked_loss[-1]:.4f}, Eval Loss Rating: {eval_rating_loss[-1]:.4f}, Eval Loss Liked: {eval_liked_loss[-1]:.4f}")

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(train_rating_loss, label='Train Rating Loss')
plt.plot(eval_rating_loss, label='Eval Rating Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Rating Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_liked_loss, label='Train Liked Loss')
plt.plot(eval_liked_loss, label='Eval Liked Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Liked Loss")
plt.legend()
plt.show()



# =================================模型评估====================================
torch.save(model.state_dict(), "model.pth")
# Initialize the model (make sure the architecture matches the saved model)
model = MultiTaskMovieLensModel(n_users, n_movies, embedding_size, hidden_size)
# Load the saved state dictionary into the model
model.load_state_dict(torch.load("model.pth"))
# Set the model to evaluation mode (important for inference)
model.eval()

def predict_and_compare(user_id, movie_id, model, user_encoder, movie_encoder, train_dataset, test_dataset):
    user_idx = user_encoder.transform([user_id])[0]
    movie_idx = movie_encoder.transform([movie_id])[0]
    example_user = torch.tensor([[user_idx]], dtype=torch.long)
    example_movie = torch.tensor([[movie_idx]], dtype=torch.long)
    example_dense_features = torch.cat((example_user, example_movie), dim=1)
    model.eval()
    with torch.no_grad():
        rating_pred, liked_pred = model(example_dense_features)
        predicted_rating = rating_pred.item()
        predicted_liked = liked_pred.item()
        actual_row = train_dataset[(train_dataset['userId'] == user_id) & (train_dataset['movieId'] == movie_id)]
    if actual_row.empty:
        actual_row = test_dataset[(test_dataset['userId'] == user_id) & (test_dataset['movieId'] == movie_id)]
    if not actual_row.empty:
        actual_rating = actual_row['rating'].values[0]
        actual_liked = actual_row['liked'].values[0]
        return {
            'User ID': user_id,
            'Movie ID': movie_id,
            'Predicted Rating': round(predicted_rating, 2),
            'Actual Rating': actual_rating,
            'Predicted Liked': 'Yes' if predicted_liked >= 0.5 else 'No',
            'Actual Liked': 'Yes' if actual_liked == 1 else 'No'
        }
    else:
        return None

example_pairs = dataset.test_df.sample(n=5)
results = []
for _, row in example_pairs.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    result = predict_and_compare(user_id, movie_id, model, user_encoder, movie_encoder, dataset.train_df, dataset.test_df)
    if result:
        results.append(result)
results_df = pd.DataFrame(results)
print(results_df.head())



# ==============================模型部署==================================
# Load your trained model (assuming it's saved as 'model.pth')
# n_users = 610  # 示例值，替换为实际用户数
# n_movies = 9724  # 示例值，替换为实际电影数
# embedding_size = 16
# hidden_size = 32
# model = MultiTaskMovieLensModel(n_users, n_movies, embedding_size, hidden_size)

model.load_state_dict(torch.load("model.pth"))
model.eval()



@serve.deployment
class ModelServeDeployment:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    async def __call__(self, request):
        json_input = await request.json()
        user_id = torch.tensor([json_input["user_id"]])
        movie_id = torch.tensor([json_input["movie_id"]])
        with torch.no_grad():
            rating_pred, liked_pred = self.model(user_id, movie_id)
        return {
            "rating_prediction": rating_pred.item(),
            "liked_prediction": liked_pred.item()
        }

# 初始化 Ray 和 Ray Serve
ray.init()
serve.start()  # 部署模型
model_deployment = ModelServeDeployment.bind(model)
serve.run(model_deployment)

# 定义服务器地址（Ray Serve 默认为 http://127.0.0.1:8000）
url = "http://127.0.0.1:8000/ModelServeDeployment"
# 示例输入
data = {
    "user_id": 123,  # 替换为实际用户 ID
    "movie_id": 456  # 替换为实际电影 ID
    }

# 向模型服务器发送 POST 请求
response = requests.post(url, json=data)
# 打印模型的响应
print(response.json())

