import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from docplex.mp.model import Model
nodes = [0, 1,2,3,4]
customers = [1,2,3,4]
weight = [0, 5, 25, 15, 15]
volume = [0, 10, 45, 35, 30]
typevehicles = [1,2,3]
capacity_weight = [20, 30, 50]  # Trọng tải của mỗi loại xe
capacity_volume = [50, 80, 100]  # Thể tích của mỗi loại xe
vendor = [1,2]
C = {
  (1,1,1): 547000,  (1,1,2): 552000,
  (1,2,1): 498000,  (1,2,2): 465000,
  (1,3,1): 557000,  (1,3,2): 577000,
  (1,4,1): 498000,  (1,4,2): 465000,
  (2,1,1): 597000,  (2,1,2): 581000,
  (2,2,1): 647000,  (2,2,2): 639000,
  (2,3,1): 607000,  (2,3,2): 617000,
  (2,4,1): 647000,  (2,4,2): 639000,
  (3,1,1): 606000,  (3,1,2): 652000,
  (3,2,1): 875000, (3,2,2): 751000,
  (3,3,1): 776000, (3,3,2): 694000,
  (3,4,1): 875000, (3,4,2): 751000}
r = {
  (1,1):100000, (1,2):100000,
  (2,1):150000, (2,2):120000,
  (3,1):200000, (3,2):170000
}
s = [
[0.00, 6.40, 22.77, 12.73, 22.07],
[5.73, 0.00, 19.72, 6.46, 20.02],
[23.34, 19.66, 0.00, 22.74, 1.58],
[11.78, 6.18, 22.49, 0.00, 22.80],
[23.63, 19.96, 1.58, 23.04, 0.00]
]
#Model declaration
m = Model(name = 'ĐỒ ÁN TỐT NGHIỆP')
x = {(i, j, k): m.binary_var(name=f'x_{i}_{j}_{k}') for i in nodes for j in nodes for k in typevehicles }
y = {(i, j, k): m.continuous_var(name=f'y_{i}_{j}_{k}') for i in nodes for j in nodes for k in typevehicles}
z = {(i, j, k): m.continuous_var(name=f'z_{i}_{j}_{k}') for i in nodes for j in nodes for k in typevehicles}
q = {(i, j, k): m.binary_var(name=f'q_{i}_{j}_{k}') for i in nodes for j in nodes for k in typevehicles }
o = {(i, j, k): m.binary_var(name=f'o_{i}_{j}_{k}') for i in nodes for j in nodes for k in typevehicles }
v = {(k,l):m.binary_var(name=f'v_{k}_{l}') for k in typevehicles for l in vendor}
d = {k: m.continuous_var(name=f'd_{k}') for k in typevehicles}
cost = {k: m.continuous_var(name=f'cost_{k}') for k in typevehicles}
c = {(k,l): m.continuous_var(name=f'c_{k}_{l}') for k in typevehicles for l in vendor}
max_sx = {k: m.continuous_var(name=f'max_sx_{k}') for k in typevehicles}
xuongca = m.integer_var_dict(typevehicles, lb=0, ub=100, name='xuongca')
a = m.integer_var_dict(typevehicles, lb=0, ub=100, name='a')
distance = {k: m.continuous_var(name=f'distance{k}') for k in typevehicles}
#----- Constraints -----
#Không quay về kho
for k in typevehicles:
  m.add_constraint(
    ct = sum(x[i,0,k] for i in nodes) ==0
  )
#--------
# M là một giá trị lớn, được sử dụng trong các ràng buộc "big-M"
M = 10000000
# Định nghĩa biến nhị phân
b1 = m.binary_var_dict(typevehicles, name='b1')
b2 = m.binary_var_dict(typevehicles, name='b2')
b3 = m.binary_var_dict(typevehicles, name='b3')
for k in typevehicles:
    # Ràng buộc để kích hoạt biến nhị phân tương ứng với mỗi khoảng của max_sx[k]
    m.add_constraints([
        max_sx[k] <= 35 + M * (1 - b1[k]),
        max_sx[k] >= 35 - M * (1 - b2[k]), max_sx[k] <= 45 + M * (1 - b2[k]),
        max_sx[k] >= 45 - M * (1 - b3[k]), max_sx[k] <= 55 + M * (1 - b3[k]),
        b1[k] + b2[k] + b3[k] == 1  # Đảm bảo chỉ một trong ba điều kiện được kích hoạt
    ])  
    # Ràng buộc để thiết lập giá trị của a dựa trên biến nhị phân
    m.add_constraint(a[k] == 8 * b1[k] + 10 * b2[k] + 16 * b3[k])
# Xương cá
for k in typevehicles:
    for i in nodes:
        for j in nodes:
            if i != j:
                m.add_constraint(ct = max_sx[k] >= s[0][j] * x[i, j, k])
    m.add_constraint(ct = max_sx[k] == sum(s[0][j] * o[i, j, k] for i in nodes for j in customers))
    m.add_constraint(ct = sum(q[i, j, k] for i in nodes for j in customers) == 1)
for i in nodes:
    for j in customers:
        for k in typevehicles:
            m.add_constraint(o[i, j, k] <= x[i, j, k])  # q_{ij}^k <= p_ij^k * x_ij^k
            m.add_constraint(o[i, j, k] <= q[i,j,k])     # q_{ij}^k <= x_ij^k
            m.add_constraint(o[i, j, k] >= x[i, j, k] - (1 - q[i,j,k]))  # q_{ij}^k >= p_ij^k - (1 - x_ij^k)     
for k in typevehicles:
    m.add_constraint(distance[k] == sum(s[i][j] * x[i, j, k] for i in nodes for j in customers))
    m.add_constraint(distance[k]<= max_sx[k] + a[k])     
#Ràng buộc (2) và (3) đảm bảo mỗi khách hàng được và chỉ được phục vụ một lần bởi một phương tiện
for k in typevehicles:
  m.add_constraint(
    ct = sum(x[0,j,k] for j in customers) <=1
  )
for j in customers:
  m.add_constraint(
    ct = sum(x[i,j,k] for k in typevehicles  for i in nodes) == 1,
    ctname = 'flow_constraint_{0!s}'
    )
for i in customers:
  m.add_constraint(
    ct = sum(x[i,j,k] for k in typevehicles for j in customers) <= 1,
    ctname = 'flow_constraint_{0!s}'.format(i)
  )
for i in nodes:
  for k in typevehicles:
    m.add_constraint(
      ct = x[i,i,k] ==0
    )
# Ràng buộc 4
for k in typevehicles:  
    for j in customers:
        m.add_constraint(ct = sum(x[i,j,k] for i in nodes) - sum(x[j,i,k] for i in nodes) >= 0)
        m.add_constraint(ct = sum(x[i,j,k] for i in nodes) - sum(x[j,i,k] for i in nodes) <= 1)
# Ràng buộc về kg
for j in customers:
    m.add_constraint(ct = sum(y[i,j,k] for i in nodes for k in typevehicles) - sum(y[j,i,k] for i in nodes for k in typevehicles) == weight[j])
for k in typevehicles:
    for j in customers:
        for i in nodes:
            if i != j:
                m.add_constraint(ct = y[i,j,k] <= (capacity_weight[k - 1] - weight[i]) * x[i,j,k])
                m.add_constraint(ct = y[i,j,k] >= weight[i]*x[i,j,k])

  # # Ràng buộc 8 và 9
  # m.capacity_cbm = []
for j in customers:
    m.add_constraint(ct = sum(z[i,j,k] for i in nodes for k in typevehicles) - sum(z[j,i,k] for i in nodes for k in typevehicles) == volume[j])
for k in typevehicles:
    for j in customers:
        for i in nodes:
          if i != j:
              m.add_constraint(ct = z[i,j,k] <= (capacity_volume[k - 1] - volume[i]) * x[i,j,k])
              m.add_constraint(ct = z[i,j,k] >= volume[i]*x[i,j,k])
# Ràng buộc hàm mục tiêu
# # 1 rớt điểm
for k in typevehicles:
  m.add_constraint(
    ct = d[k] == sum(x[i,j,k] for i in nodes for j in customers) - 1*sum(o[i,j,k] for i in nodes for j in customers)
  )
####Phí chính
for k in typevehicles:
  m.add_constraint(
    ct = sum(v[k,l] for l in vendor) == 1*sum(o[i,j,k] for i in nodes for j in customers)
  )
  for l in vendor:
    for i in nodes:
      for j in customers:
        m.add_constraint(
          ct = c[k,l]>= v[k,l]*C[k,j,l]- M*(1-x[i,j,k])
        )
#----- Objective Function -----
#Minimize the total cost
object_model = sum(d[k]*r[k,l]*v[k,l] for k in typevehicles for l in vendor) + sum(c[k,l] for k in typevehicles for l in vendor)+ sum(distance[k] for k in typevehicles)
m.minimize(object_model)
a = m.solve(log_output=True)
print(a)
for k in typevehicles:
    # Tính tổng quãng đường của từng chuyến
    total_distance_k = sum(s[i][j] * x[i, j, k].solution_value for i in nodes for j in nodes if i != j)
    # Khoảng cách từ kho đến điểm xa nhất cho xe k
    max_distance_from_depot_k = max_sx[k].solution_value
    
    print(f"Vehicle type {k}:")
    print(f"  Total distance traveled: {total_distance_k}")
    print(f"  Max distance from depot: {max_distance_from_depot_k}")
print(object_model)
for (k, l), var in v.items():
    # Kiểm tra xem biến var đã có giá trị hay chưa (có được giải quyết trong mô hình hay không)
    if var.solution_value ==1:
        print(f"Xe và nhà thầu được chọn: Xe {k} và nhà thầu {l}")
m.print_solution()