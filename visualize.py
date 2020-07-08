import matplotlib.pyplot as plt


def visualize(result):
	colors = ('steelblue', 'darkorange', 'mediumseagreen')
	labels = ('S(t)', 'I(t)', 'R(t)')
	markers = ('^', 's', 'o')
	plt.switch_backend("Qt5Agg")
	plt.rcParams["xtick.major.pad"] = 8
	plt.rcParams["axes.xmargin"] = 0
	plt.rcParams["axes.ymargin"] = 0
	plt.figure(figsize = (18, 12))
	for i, (l, c, m) in enumerate(zip(labels, colors, markers)):
		plt.plot(range(result.shape[0]), result[:, i], color = c)
		plt.scatter(range(result.shape[0]), result[:, i], marker = m)
		plt.plot([], [], f'-{m}', color = c, label = l)
	plt.xlabel('Day', labelpad = 10)
	plt.xticks(range(result.shape[0]), range(result.shape[0]), rotation=30)
	plt.ylabel('# of people', labelpad = 10)
	plt.title('SIR Model', pad = 15)
	plt.legend(loc = 'best')
	plt.savefig("plot.png")
	plt.show()