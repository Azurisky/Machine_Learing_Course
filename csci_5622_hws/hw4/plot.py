import matplotlib.pyplot as plt


C = [1, 2, 3, 4, 5, 6]
# C = [1, 10, 100, 1000]

linear=[0.965, 0.958, 0.948, 0.946]
poly_2=[0.944, 0.972, 0.983, 0.990, 0.988, 0.988]
poly_3=[0.511, 0.944, 0.974, 0.987, 0.988, 0.988]
poly_4=[0.507, 0.518, 0.923, 0.964, 0.984, 0.986]
poly_5=[0.507, 0.507, 0.535, 0.886, 0.949, 0.980]
poly_6=[0.507, 0.507, 0.507, 0.553, 0.850, 0.924]
RBF_1 = [0.972, 0.973, 0.973, 0.973]
RBF_2 = [0.960, 0.972, 0.980, 0.983]
RBF_3 = [0.934, 0.958, 0.968, 0.969]
RBF_4 = [0.517, 0.934, 0.957, 0.966]


# plt.plot(C, linear, label="Linear")

plt.plot(C, poly_2, label="poly (p = 2)")
plt.plot(C, poly_3, label="poly (p = 3)")
plt.plot(C, poly_4, label="poly (p = 4)")
plt.plot(C, poly_5, label="poly (p = 5)")
plt.plot(C, poly_6, label="poly (p = 6)")

# plt.plot(C, RBF_1, label="RBF (r = 0.1)")
# plt.plot(C, RBF_2, label="RBF (r = 0.01)")
# plt.plot(C, RBF_3, label="RBF (r = 0.001)")
# plt.plot(C, RBF_4, label="RBF (r = 0.0001)")

plt.xticks(range(7), ('0', '1', '10', '100', '1000', '10000', '100000'))
plt.xlabel('C')
plt.ylabel('Accuracy score')
plt.legend()
plt.show()