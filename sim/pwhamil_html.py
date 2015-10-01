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

if __name__ == '__main__':
	#labelTrials = pickle.load( open( "labelTrials.p", "rb" ) )

	labels1 = ['control', 'mass', 'inertia']
	labels2 = ['control-diag', 'mass-diag', 'inertia-diag']
	#labels3 = ['control-raw', 'mass-raw', 'inertia-raw']
	labels3 = ['control-estimates', 'mass-estimates', 'inertia-estimates']
	labels4 = ['control-150903', 'mass-150903', 'inertia-150903']

	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite'), labels1)
	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite-diag'), labels2)
	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite-estimates'), labels3)
	createHtml(os.path.join(os.getcwd(),'PWHamil','_composite-150903'), labels4)
