import matplotlib.pyplot as plt

epochs = list(range(10))  # epochs 0 to 9

# Metrics values
auc_values = [0.8635014386378906, 0.8905978657114924, 0.9046539632351012, 0.9121045912557028, 0.9166272890132912,
              0.920585900963621, 0.9228825729058805, 0.925305268271533, 0.9276784571624139, 0.9287887834790753]

f1_values = [0.8591291887526792, 0.8727928669576219, 0.8788311238773068, 0.8828969522461629, 0.8846587137750659,
             0.8851125563401675, 0.888095544800427, 0.888258516237958, 0.8910685805422648, 0.8924805276304618]

recall_values = [0.8946383409205867, 0.9116590794132524, 0.9150101163378856, 0.9150733434496712, 0.9185381891755184,
                0.9026681841173495, 0.915402124430956, 0.9047926150733434, 0.9184496712190188, 0.9171851289833081]

precision_values = [0.8263312192672017, 0.8371050706547613, 0.8454043065275555, 0.8529065107727123, 0.8531895650539718,
                    0.8682267657540411, 0.8623708945355777, 0.8723178581878475, 0.865272813914701, 0.8690718684847468]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(epochs, auc_values, marker='o', linestyle='-', color='b', label='AUC')
plt.plot(epochs, f1_values, marker='o', linestyle='-', color='g', label='F1 Score')
plt.plot(epochs, recall_values, marker='o', linestyle='-', color='r', label='Recall')
plt.plot(epochs, precision_values, marker='o', linestyle='-', color='c', label='Precision')

plt.title('Metrics Progression over Epochs on SAKT')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.xticks(epochs)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
