import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 用于显示训练进度的库
import numpy as np
class LocationEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LocationEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, seq):
        embeds = self.embeddings(seq)
        lstm_out, _ = self.lstm(embeds.view(len(seq), 1, -1))
        output = self.linear(lstm_out.view(len(seq), -1))
        output_scores = self.softmax(output)
        return output_scores

def train_model(train_data, vocab_size, embedding_dim=400, hidden_dim=128, num_epochs=100, lr=0.01, device='gpu'):
    model = LocationEmbeddingModel(vocab_size, embedding_dim, hidden_dim).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        for current, next_ in train_data:
            current, next_ = current.to(device), next_.to(device)  # 将数据移动到指定设备
            model.zero_grad()
            log_probs = model(current)
            loss = loss_function(log_probs, next_)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'location_embedding_model.pth')
    return model

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    train_data = torch.load('train_data.pt')
    vocab_size = torch.load('vocab_size.pt')
    model = train_model(train_data, vocab_size, device=device)  # 使用指定设备进行训练

    # 提取嵌入向量
    location_embeddings = model.embeddings.weight.data.cpu().numpy()  # 如果需要，将数据移动到CPU

    # 保存嵌入向量为NumPy数组
    np.save('location_embeddings.npy', location_embeddings)
    print(location_embeddings)
