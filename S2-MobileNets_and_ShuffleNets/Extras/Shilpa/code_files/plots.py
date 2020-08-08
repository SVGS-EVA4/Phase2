import matplotlib.pyplot as plt

def plot_train_test_lr(test=[], train=[],l_rate=[], Type=''):
  fig, ax1 = plt.subplots()

  ax1.set_xlabel('epoch (s)')
  ax1.set_ylabel(Type, color='g')
  testline, = ax1.plot( test, color='g')
  ax1.tick_params(axis='y', labelcolor='g')

  trainline, = ax1.plot( train, color='r')
  ax1.legend((trainline, testline), ('Train', 'Test'), loc=7)
  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


  ax2.set_ylabel('learning rate', color='b')  # we already handled the x-label with ax1
  lrline, = ax2.plot( l_rate, color='b')
  ax2.legend((lrline, ), ('LR',), loc=8)
  ax2.tick_params(axis='y', labelcolor='b')

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  title= f"Learning Rate and Train/Test {Type} Comparison"
  plt.title(title)
  plt.show()

def plot_graphs(stats=[],labels=[],xlabel='',ylabel='',plot_title=''):
  plt.figure(figsize=(10, 5))
  ax = plt.subplot(111)
  for i in range(len(labels)):
    ax.plot(stats[i],label=labels[i])

  ax.set(title=plot_title, xlabel=xlabel, ylabel=ylabel)
  ax.legend()
  plt.show()
