import matplotlib.pyplot as plt


line_up, = plt.plot([1, 2, 3, 4], [6, 7, 8, 9], label='Test Line')

plt.legend(handles=[line_up], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.show()
