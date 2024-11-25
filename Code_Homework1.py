import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Bước 1: Tạo dữ liệu
np.random.seed(42)  # Để tái hiện kết quả
x = np.linspace(0, 10, 100).reshape(-1, 1)  # Biến đầu vào trong khoảng [0, 10]
epsilon = np.random.normal(0, np.sqrt(4), x.shape)  # Nhiễu Gaussian với phương sai 4
y = 2 * x + epsilon  # Biến đầu ra

# Bước 2: Hồi quy tuyến tính
linear_model = LinearRegression()
linear_model.fit(x, y)
y_linear_pred = linear_model.predict(x)

# Bước 3: Hồi quy đa thức bậc 9
poly = PolynomialFeatures(degree=9)
x_poly = poly.fit_transform(x)  # Biến đổi x thành đặc trưng bậc 9
poly_model = LinearRegression()
poly_model.fit(x_poly, y)
y_poly_pred = poly_model.predict(x_poly)

# Bước 4: Tính toán sai số bình phương trung bình (MSE)
mse_linear = mean_squared_error(y, y_linear_pred)
mse_poly = mean_squared_error(y, y_poly_pred)

# Bước 5: Biểu đồ so sánh
plt.figure(figsize=(14, 7))

# Biểu đồ dữ liệu gốc
plt.scatter(x, y, color='gray', alpha=0.5, label="Dữ liệu (có nhiễu)")

# Hồi quy tuyến tính
plt.plot(x, y_linear_pred, color='blue', label="Hồi quy tuyến tính")

# Hồi quy bậc 9
plt.plot(x, y_poly_pred, color='red', linestyle='--', label="Hồi quy bậc 9")

# Gắn nhãn và hiển thị
plt.title("So sánh Hồi quy tuyến tính và đa thức bậc 9")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# In kết quả
print("Hồi quy tuyến tính:")
print(f"Hệ số góc (slope): {linear_model.coef_[0][0]:.4f}")
print(f"Hệ số chặn (intercept): {linear_model.intercept_[0]:.4f}")
print(f"MSE: {mse_linear:.4f}")

print("\nHồi quy bậc 9:")
print(f"MSE: {mse_poly:.4f}")
