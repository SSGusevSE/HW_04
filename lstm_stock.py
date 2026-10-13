import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ============================================================
# 1. Загрузка данных
# ============================================================
ticker = "AAPL"
df = yf.download(ticker, start="2018-01-01", end="2026-02-01", auto_adjust=True, progress=False)
prices = df["Close"].dropna()

# Приводим к одномерному массиву
if hasattr(prices, "columns"):
    prices = prices.iloc[:, 0]

print(f"Всего данных: {len(prices)} дней")
print(f"Период: {prices.index[0].date()} — {prices.index[-1].date()}")

# ============================================================
# 2. Разделение: обучение до 2025-12-31, тест — январь 2026
# ============================================================
train_prices = prices[:"2025-12-31"]
january_prices = prices["2026-01-01":"2026-01-31"]

print(f"\nОбучающая выборка: {len(train_prices)} дней")
print(f"Январь 2026 (реальные данные): {len(january_prices)} дней")
print(f"Первые 10 торговых дней января:")
print(january_prices.head(10))

# ============================================================
# 3. Нормализация (Z-score по train)
# ============================================================
mean = train_prices.mean()
std = train_prices.std()

scaled_train = (train_prices.to_numpy().reshape(-1) - mean) / std
scaled_jan = (january_prices.to_numpy().reshape(-1) - mean) / std

# ============================================================
# 4. Создание окон
# ============================================================
WINDOW = 20

def make_windows(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), \
           torch.tensor(np.array(y), dtype=torch.float32)

X_train, y_train = make_windows(scaled_train, WINDOW)

# 80/20 split внутри train для валидации
split = int(len(X_train) * 0.8)
X_tr, y_tr = X_train[:split], y_train[:split]
X_val, y_val = X_train[split:], y_train[split:]

print(f"\nОкна: train={len(X_tr)}, val={len(X_val)}")

# ============================================================
# 5. Baseline — предсказание = последнее значение окна
# ============================================================
baseline_pred = X_val[:, -1]
baseline_mse = nn.MSELoss()(baseline_pred, y_val).item()
print(f"\nBaseline MSE (val): {baseline_mse:.6f}")

# ============================================================
# 6. MLP модель
# ============================================================
class PriceMLP(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

mlp_model = PriceMLP(WINDOW)
mlp_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

print("\n--- Обучение MLP ---")
for epoch in range(30):
    mlp_model.train()
    for xb, yb in mlp_loader:
        optimizer.zero_grad()
        loss = criterion(mlp_model(xb), yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"  Эпоха {epoch+1}, loss={loss.item():.6f}")

mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_model(X_val)
mlp_mse = criterion(mlp_pred, y_val).item()
print(f"MLP MSE (val): {mlp_mse:.6f}")

# ============================================================
# 7. LSTM модель
# ============================================================
class PriceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

lstm_model = PriceLSTM()
X_tr_lstm = X_tr.unsqueeze(-1)
X_val_lstm = X_val.unsqueeze(-1)
lstm_loader = DataLoader(TensorDataset(X_tr_lstm, y_tr), batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

print("\n--- Обучение LSTM ---")
for epoch in range(50):
    lstm_model.train()
    for xb, yb in lstm_loader:
        optimizer.zero_grad()
        loss = criterion(lstm_model(xb), yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"  Эпоха {epoch+1}, loss={loss.item():.6f}")

lstm_model.eval()
with torch.no_grad():
    lstm_pred = lstm_model(X_val_lstm)
lstm_mse = criterion(lstm_pred, y_val).item()
print(f"LSTM MSE (val): {lstm_mse:.6f}")

# ============================================================
# 8. Сравнение моделей на валидации
# ============================================================
print("\n" + "=" * 50)
print("СРАВНЕНИЕ МОДЕЛЕЙ (валидация)")
print("=" * 50)
print(f"  Baseline MSE: {baseline_mse:.6f}")
print(f"  MLP MSE:      {mlp_mse:.6f}")
print(f"  LSTM MSE:     {lstm_mse:.6f}")

# ============================================================
# 9. Авторегрессионное предсказание на январь (10 дней)
# ============================================================
PREDICT_DAYS = 10

# Берём последние WINDOW дней декабря
last_window = torch.tensor(scaled_train[-WINDOW:], dtype=torch.float32)

future_lstm = []
lstm_model.eval()
with torch.no_grad():
    window = last_window.clone()
    for _ in range(PREDICT_DAYS):
        x = window.reshape(1, WINDOW, 1)
        pred = lstm_model(x)
        future_lstm.append(pred.item())
        window = torch.cat([window[1:], pred.flatten()])

# Обратная денормализация
pred_real = np.array(future_lstm) * std + mean
actual_real = january_prices.head(PREDICT_DAYS).values.reshape(-1)
actual_dates = january_prices.head(PREDICT_DAYS).index

print("\n" + "=" * 50)
print(f"ПРЕДСКАЗАНИЕ vs РЕАЛЬНОСТЬ (первые {PREDICT_DAYS} дней января 2026)")
print("=" * 50)
print(f"{'Дата':<14} {'Предсказание':>14} {'Реальная цена':>14} {'Ошибка':>10}")
print("-" * 54)
for i in range(min(PREDICT_DAYS, len(actual_real))):
    d = actual_dates[i].strftime("%Y-%m-%d")
    err = pred_real[i] - actual_real[i]
    print(f"{d:<14} {pred_real[i]:>14.2f} {actual_real[i]:>14.2f} {err:>+10.2f}")

# Метрики
n = min(PREDICT_DAYS, len(actual_real))
mae = np.mean(np.abs(pred_real[:n] - actual_real[:n]))
rmse = np.sqrt(np.mean((pred_real[:n] - actual_real[:n]) ** 2))
mape = np.mean(np.abs((pred_real[:n] - actual_real[:n]) / actual_real[:n])) * 100

print(f"\nМетрики предсказания:")
print(f"  MAE:  {mae:.2f} $")
print(f"  RMSE: {rmse:.2f} $")
print(f"  MAPE: {mape:.2f} %")

# ============================================================
# 10. Визуализация
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# График 1: Последние 60 дней + предсказание
last_n = 60
recent = train_prices.tail(last_n)
ax = axes[0]
ax.plot(recent.index, recent.values, label="Обучающие данные (декабрь)", color="blue")
ax.plot(actual_dates[:n], actual_real[:n], label="Реальная цена (январь)", color="black", marker="o")
ax.plot(actual_dates[:n], pred_real[:n], label="Предсказание LSTM", color="red", marker="x", linestyle="--")
ax.set_title(f"{ticker}: Предсказание на январь 2026")
ax.set_ylabel("Цена ($)")
ax.legend()
ax.grid(True, alpha=0.3)

# График 2: Сравнение моделей на валидации
ax2 = axes[1]
val_indices = range(len(y_val))
ax2.plot(val_indices, y_val.numpy() * std + mean, label="Реальная цена", color="black", alpha=0.7)
ax2.plot(val_indices, baseline_pred.numpy() * std + mean, label=f"Baseline (MSE={baseline_mse:.4f})", alpha=0.5)
ax2.plot(val_indices, mlp_pred.numpy() * std + mean, label=f"MLP (MSE={mlp_mse:.4f})", alpha=0.5)
ax2.plot(val_indices, lstm_pred.numpy() * std + mean, label=f"LSTM (MSE={lstm_mse:.4f})", color="red", alpha=0.5)
ax2.set_title("Сравнение моделей на валидационной выборке")
ax2.set_ylabel("Цена ($)")
ax2.set_xlabel("Индекс")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lstm_results.png", dpi=150)
print("\nГрафик сохранён: lstm_results.png")
plt.show()
