import matplotlib.pyplot as plt

def plot_ttloss_lr:
	fig, ax1 = plt.subplots()

	ax1.set_xlabel('epoch (s)')
	ax1.set_ylabel('accuracy', color='g')
	testline, = ax1.plot( test_acc, color='g')
	ax1.tick_params(axis='y', labelcolor='g')

	trainline, = ax1.plot( train_acc_epoch_end, color='r')
	ax1.legend((trainline, testline), ('Train', 'Test'), loc=7)
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


	ax2.set_ylabel('learning rate', color='b')  # we already handled the x-label with ax1
	lrline, = ax2.plot( l_rate, color='b')
	ax2.legend((lrline, ), ('LR',), loc=8)
	ax2.tick_params(axis='y', labelcolor='b')

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.title("Learning Rate and Train/test Accuracy Comparison")
	plt.show()