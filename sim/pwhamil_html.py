import pickle
import os

def createHtml(pageName, labels):
	html = ['<html>\n<table>\n']

	for label in labels:

		html.append('<tr>\n')
		#html.append('<td> <span style="font-size:50px;"> %s </span> </td>\n' % (str(labelTrials[label])))

		for column in ['ddx', 'ddy', 'ddtheta']:
			html.append('<td><img src="%s-%s.svg"></img></td>\n' % (str(label), column))

		html.append('</tr>\n')

	html.append('</table>\n</html>')

	open(pageName + '.html','w').write('\n'.join(html))

	return html

def createHtml2(pageName, labels):
	html = ['<html>\n<table>\n']

	for label in labels:

		html.append('<tr>\n')
		#html.append('<td> <span style="font-size:50px;"> %s </span> </td>\n' % (str(labelTrials[label])))

		for column in ['ddx', 'ddy', 'ddtheta']:
			html.append('<td><img src="%s-%s.svg"></img></td>\n' % (str(label), column))

		html.append('</tr>\n')

		html.append('<tr>\n')
		#html.append('<td> <span style="font-size:50px;"> %s </span> </td>\n' % (str(labelTrials[label])))

		for column in ['x', 'y', 'theta']:
			html.append('<td><img src="%s-%s.svg"></img></td>\n' % (str(label), column))

		html.append('</tr>\n')

	html.append('</table>\n</html>')

	open(pageName + '.html','w').write('\n'.join(html))

	return html

if __name__ == '__main__':
	#labelTrials = pickle.load( open( "labelTrials.p", "rb" ) )

	labels1 = ['control', 'mass', 'inertia']
	labels2 = ['control-diag', 'mass-diag', 'inertia-diag']
	#labels3 = ['control-raw', 'mass-raw', 'inertia-raw']
	labels3 = ['control-estimates', 'mass-estimates', 'inertia-estimates']
	labels4 = ['control-150903', 'mass-150903', 'inertia-150903']
	labels5 = ['151118-control', '151118-mass', '151118-inertia']
	labels6 = ['151118-control-estimates', '151118-mass-estimates', '151118-inertia-estimates']
	labels7 = ['151118-control-shai-estimates', '151118-mass-shai-estimates', '151118-inertia-shai-estimates']
	labels8 = ['151118-control-shaiIntegrate-estimates', '151118-mass-shaiIntegrate-estimates', '151118-inertia-shaiIntegrate-estimates']
	labels9 = ['151118-control-shaiDerivative-estimates', '151118-mass-shaiDerivative-estimates', '151118-inertia-shaiDerivative-estimates']

	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite'), labels1)
	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite-diag'), labels2)
	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite-estimates'), labels3)
	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite-150903'), labels4)
	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite-151118'), labels5)
	createHtml2(os.path.join(os.getcwd(),'PWHamil','_composite-estimates-151118'), labels6)
	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite-shai-151118'), labels7)
	createHtml2(os.path.join(os.getcwd(),'PWHamil','_composite-shaiIntegrate-151118'), labels8)
	createHtml2(os.path.join(os.getcwd(),'PWHamil','_composite-shaiDerivative-151118'), labels9)
