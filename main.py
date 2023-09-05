import pandas as pd
import torch
import train  # 假设 train.py 是你的训练模型的代码文件
from scipy.spatial.distance import jensenshannon  # Jensen-Shannon 散度
import folium
from heapq import nlargest

def load_and_preprocess_data():
    df = pd.read_csv("data.csv")
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    df.sort_values(by=['DriveNo', 'Date and Time'], inplace=True)
    trajectories = {}
    for drive_no in df['DriveNo'].unique():
        drive_data = df[df['DriveNo'] == drive_no]
        locations = drive_data[['Latitude', 'Longitude']].values.tolist()
        trajectories[drive_no] = locations
    location_pairs = []
    for drive_no, locations in trajectories.items():
        for i in range(len(locations) - 1):
            current_location = tuple(locations[i])
            next_location = tuple(locations[i + 1])
            location_pairs.append((current_location, next_location))
    unique_locations = list(set([loc for pair in location_pairs for loc in pair]))
    location_to_index = {loc: index for index, loc in enumerate(unique_locations)}
    index_to_location = {index: loc for loc, index in location_to_index.items()}
    index_pairs = [(location_to_index[current], location_to_index[next_]) for current, next_ in location_pairs]
    train_data = [(torch.tensor([current]), torch.tensor([next_])) for current, next_ in index_pairs]
    torch.save(train_data, 'train_data.pt')
    torch.save(len(unique_locations), 'vocab_size.pt')
    return unique_locations, location_to_index, index_to_location

def load_model(vocab_size, embedding_dim, hidden_dim, device):
    model = train.LocationEmbeddingModel(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load('location_embedding_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_predictions(model, vocab_size, unique_locations, device):
    predictions = {}
    for index in range(vocab_size):
        input_tensor = torch.tensor([index]).to(device)
        with torch.no_grad():
            log_probs = model(input_tensor)
            output_probs = torch.exp(log_probs).squeeze().cpu().numpy()
        predictions[unique_locations[index]] = output_probs
    return predictions

def find_top_n_similar_locations(target_location, n, unique_locations, predictions):
    similarities = {}
    for loc in unique_locations:
        if loc != target_location:
            p = predictions[loc]
            q = predictions[target_location]
            js_divergence = jensenshannon(p, q)
            similarity_score = 1 / (1 + js_divergence)
            similarities[loc] = similarity_score
    top_n_similar = nlargest(n, similarities, key=similarities.get)
    return top_n_similar


def predict_next_location(current_location, model, location_to_index, index_to_location, device):
    model.eval()
    current_index = torch.tensor([location_to_index[current_location]]).to(device)
    with torch.no_grad():
        output_probs = model(current_index).exp().cpu()
        next_index = output_probs.argmax().item()
    next_location = index_to_location[next_index]
    return next_location

def get_next_actual_location(target_location, trajectories):
    for drive_no, locations in trajectories.items():
        for i in range(len(locations) - 1):
            if tuple(locations[i]) == target_location:
                return tuple(locations[i + 1])
    return None


def create_advanced_map(target_location, top_n_similar_locations, model, location_to_index, index_to_location, device):
    m = folium.Map(location=target_location[::-1], zoom_start=13)

    # 目标位置（红色）
    folium.CircleMarker(
        location=target_location[::-1],
        radius=6,
        color='red',
        fill=True,
        fill_opacity=0.6,
    ).add_to(m)

    # 与目标位置相似的位置（绿色）
    for loc in top_n_similar_locations:
        folium.CircleMarker(
            location=loc[::-1],
            radius=10,
            color='green',
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)

        # 预测下一个位置并用蓝色线连接
        next_loc = predict_next_location(loc, model, location_to_index, index_to_location, device)
        folium.PolyLine([loc[::-1], next_loc[::-1]], color="blue", weight=2.5, opacity=1).add_to(m)

        # 标记预测的下一个位置（蓝色）
        folium.CircleMarker(
            location=next_loc[::-1],
            radius=4,
            color='blue',
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)

    # 预测目标位置的下一个位置并用红色线连接
    next_target_loc = predict_next_location(target_location, model, location_to_index, index_to_location, device)
    folium.PolyLine([target_location[::-1], next_target_loc[::-1]], color="red", weight=2.5, opacity=1).add_to(m)

    # 标记预测的下一个目标位置（粉色）
    folium.CircleMarker(
        location=next_target_loc[::-1],
        radius=4,
        color='pink',
        fill=True,
        fill_opacity=0.6,
    ).add_to(m)

    return m

if __name__ == "__main__":
    print(torch.cuda.is_available())


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    unique_locations, location_to_index, index_to_location = load_and_preprocess_data()
    vocab_size = len(unique_locations)
    embedding_dim = 400
    hidden_dim = 128
    model = load_model(vocab_size, embedding_dim, hidden_dim, device)
    predictions = generate_predictions(model, vocab_size, unique_locations, device)
    target_location = unique_locations[1]
    top_5_similar_locations = find_top_n_similar_locations(target_location, 3, unique_locations, predictions)
    # 创建增强的地图
    m = create_advanced_map(target_location, top_5_similar_locations, model, location_to_index, index_to_location,device)
    m.save("advanced_map.html")
