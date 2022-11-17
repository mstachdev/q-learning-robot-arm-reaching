import matplotlib.pyplot as plt

learning_rate = 0.7
f = "saved_q_table_lr0_" + str(learning_rate[-1:]) + ".pickle"
with open(f, 'rb') as handle:
    ql_runs = pickle.load(handle)

x = range(0, len(ql_runs[0])) # 500
y = np.average(np.asarray(ql_runs), axis=0)

plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title(f"QLearning - learning rate {0}".format(learning_rate))
plt.ylim(-10000, 0)
plt.plot(x,y)
