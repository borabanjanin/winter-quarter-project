obs = ['acc','x','y','dx','dy','theta','omega','v','delta']

mdls = ['LLStoPuck','Puck','LLSPersist']

tmts = ['c','m','i']

for mdl in mdls:

  html = ['<html>\n<table>\n']

  for ob in obs:
    html.append('<tr>\n')

    html.append('<td><img src="%s-t_%s.png"></img></td>\n'
                % ('_'.join(tmts)+'_d',ob))

    for _ in (['_'.join(tmts)] + [tmt+'_o' for tmt in tmts]):
      html.append('<td><img src="%s-%s-t_%s.png"></img></td>\n'
                  % (mdl,_,ob))
     
    html.append('</tr>\n')


  html.append('</table>\n</html>')

  open('_%s.html'%mdl,'w').write('\n'.join(html))
