import matplotlib.pyplot as plt
import pickle

ex_num = 3
num_pts = 1000
dir = f'/Users/brittany/Documents/GitHub/rigorous_ML/output/Leslie/23.5_23.5/{num_pts}/{ex_num}'
train_log_file = dir + f'/logs/train_losses_.pkl'
test_log_file = dir + f'/logs/test_losses_.pkl'

for file in [train_log_file, test_log_file]:
    with open(file, 'rb') as f:
        logs = pickle.load(f)

    print(logs.keys())
    plt.yscale('log')
    plt.plot(logs['loss_total'], label = 'loss_total')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.close()