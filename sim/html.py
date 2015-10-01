import pickle
def createHtml(pageName, labels, labelTrials):
	html = ['<html>\n<table>\n']

	for label in labels:

		html.append('<tr>\n')
		html.append('<td> <span style="font-size:50px;"> %s </span> </td>\n' % (str(labelTrials[label])))

		for i in range(1, 16):
			html.append('<td><img src="%s-%s.svg"></img></td>\n' % (str(label), i))

		html.append('</tr>\n')

	html.append('</table>\n</html>')

	open(pageName + '.html','w').write('\n'.join(html))

	return html

if __name__ == '__main__':
	labelTrials = pickle.load( open( "labelTrials.p", "rb" ) )

	labels1 = ['control', 'mass', 'inertia']
	labels2 = ['control', 'c-2', 'c-4', 'c-6', 'c-7', 'c-8', 'c-9']
	labels3 = ['mass', 'm-2', 'm-3', 'm-4', 'm-5', 'm-6', 'm-7', 'm-8', 'm-9']
	labels4 = ['inertia', 'i-1', 'i-2', 'i-4', 'i-6', 'i-7', 'i-8', 'i-9']
	labels5 = ['control2', 'mass2', 'inertia2']

	labels6 = ['220715-control', '220715-mass', '220715-inertia']
	labels7 = ['220715-control', '220715-c-2', '220715-c-4', '220715-c-6', '220715-c-7', '220715-c-8', '220715-c-9']
	labels8 = ['220715-mass', '220715-m-2', '220715-m-3', '220715-m-4', '220715-m-5', '220715-m-6', '220715-m-7', '220715-m-8', '220715-m-9']
	labels9 = ['220715-inertia', '220715-i-1', '220715-i-2', '220715-i-4', '220715-i-6', '220715-i-7', '220715-i-8', '220715-i-9']

	labels10 = ['250715-control', '250715-mass', '250715-inertia']
	labels11 = ['250715-control', '250715-c-2', '250715-c-4', '250715-c-6', '250715-c-7', '250715-c-8', '250715-c-9']
	labels12 = ['250715-mass', '250715-m-2', '250715-m-3', '250715-m-4', '250715-m-5', '250715-m-6', '250715-m-7', '250715-m-8', '250715-m-9']
	labels13 = ['250715-inertia', '250715-i-1', '250715-i-2', '250715-i-4', '250715-i-6', '250715-i-7', '250715-i-8', '250715-i-9']


	#createHtml('composite', labels1, labelTrials)
	#createHtml('control', labels2, labelTrials)
	#createHtml('mass', labels3, labelTrials)
	#createHtml('inertia', labels4, labelTrials)
	#createHtml('composite2', labels5, labelTrials)

	createHtml('_220715-composite', labels6, labelTrials)
	createHtml('_220715-control', labels7, labelTrials)
	createHtml('_220715-mass', labels8, labelTrials)
	createHtml('_220715-inertia', labels9, labelTrials)

	createHtml('_250715-composite', labels10, labelTrials)
	createHtml('_250715-control', labels11, labelTrials)
	createHtml('_250715-mass', labels12, labelTrials)
	createHtml('_250715-inertia', labels13, labelTrials)
