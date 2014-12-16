import os

import numpy as np
import pylab as plt

from glob import glob

try:
  from shrevz import tk
  from shrevz import util as shutil
except:
  tk = None

font={'family':'sans-serif',
      'sans-serif':['Computer Modern Roman'],
      'size':16}

def set_font():
  from matplotlib import rc
  rc('font',**font)
  rc('text', usetex=True)

#def load(tmt):
#  T = glob(os.path.join('*_'+tmt,base+'.out'))

#  remap = dict(relative1='relative',constant1='constant',individualized1='personal')

#  D = dict()
#  for fi in T:
#    L = [s.strip().split(',') for s in open(fi).readlines()]
#    for l in L:
#      if base == 'trial':
#        subj = l[0]; sch = remap[l[1]]; d = [float(l[2]),float(l[3]),float(l[4]),float(l[5])]
#      else:
#        subj = l[0]; sch = remap[l[1]]; d = [float(l[2]),float(l[3])]
#      if subj not in D.keys():
#        D[subj] = dict()
#      if sch not in D[subj].keys():
#        D[subj][sch] = []
#      D[subj][sch].append(d)

#  # WARNING -- assumes every subject used every schme
#  Subj = D.keys(); Sch = D[subj].keys() 
#  A = dict();
#  for sch in Sch:
#    A[sch] = []
#  for subj in Subj:
#    for sch in Sch:
#      D[subj][sch] = np.asarray(D[subj][sch])
#      A[sch].extend(D[subj][sch])

#  for sch in Sch:
#    A[sch] = np.asarray(A[sch])

#  Sch = ['constant','relative','personal']

#  return A,D,Subj,Sch


def hist(label,classes,data,fig=None,ax=None,
         colors=None,lines=None,dashes=None,
         xlim=None,lw=8.,bins=10,sigma=1.,
         add_bars=True,add_lines=True,xlabel=True,ylabel=True,pvals=[]):
  """
  Customizable histogram for multiple classes

  WARNING: for more than 3 classes, must specify colors, lines, and dashes 

  Inputs
    label - str
    classes - list - data classes
    data - dict - histogram data for each class

    (optional for 3 or fewer classes)
    colors - dict - color for each class
    lines - dict - line style for each class
    dashes - dict - dashes for each class

    (optional)
    fig, ax - figure and axis handles
    add_bars - bool - histogram bar plot
    add_lines - bool - smoothed histogram line plot
    xlim - (x,X) - x-axis limits
    lw - float - line width
    bins - int - # of bins in histogram
    sigma - float - convolution kernel width
    xlabel,ylabel - str - labels for x- and y- axes
    lbls - list of (a,b,lbl) - add label to lines connecting classes a & b

  Output
    fig, ax - figure and axis handles

  Effects
    adds histogram data (bars or lines) to axis
  """
  if fig is None:
    fig = plt.figure(); plt.clf()
  if ax is None:
    ax = plt.subplot(111); ax.grid('on')
  if colors is None:
    colors = {}
    colors_ = ['#aa3e39','#542b72','#255c69'] # magenta, purple, cyan
    for j,cla in enumerate(classes):
      colors[cla] = colors_[j]
  if lines is None:
    lines = {}
    lines_  = ['-.','--','-']
    for j,cla in enumerate(classes):
      lines[cla] = lines_[j]
  if dashes is None:
    dashes = {}
    dashes_ = [[12,4,4,4],[16,4],[]]
    for j,cla in enumerate(classes):
      dashes[cla] = dashes_[j]

  def add_pval(ax,i,j,text,X,Y,yoffs=0.,col='k'):
    x = (X[i]+X[j])/2
    y = (0.95 + yoffs)*max(Y[i], Y[j]) + yoffs
    dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-','color':col,
                 'shrinkA':25,'shrinkB':25,'lw':lw/2.}
    ax.annotate(text,xy=(x,y+.6),zorder=10,ha='center',bbox=dict(fc='w',ec='k',lw=lw/4.,pad=lw,color=col),color=col)
    ax.annotate('',xy=(X[i],y),xytext=(X[j],y),arrowprops=props)

  ymax = -np.inf
  X = {}; Y = {}
  for cla in classes:
    line = lines[cla]; dash = dashes[cla]; color = colors[cla]
    d = data[cla]
    X[cla] = d.mean()
    if add_bars:
      n,_,_ = ax.hist(d,bins=10,color=color,range=xlim,normed=False,alpha=.5,label=cla)
      ymax = max([ymax,n.max()])
    if add_lines:
      x,d = shutil.densityPlot1(d,bins=bins*10,returnX=True,sigma=sigma,boundary=xlim)
      ymax = max([ymax,d.max()])
      Y[cla] = d.max()
      if add_bars:
        ax.plot(x,d,color='k',lw=lw)
        ax.plot(x,d,color=color,lw=.5*lw,label=cla,linestyle=line)
      else:
        ax.plot(x,d,color=color,lw=lw,label=cla,linestyle=line,dash_joinstyle='bevel',dash_capstyle='butt',dashes=dash)
    ax.set_xlim(xlim)
    if pvals:
      ax.set_ylim((0.,1.5*ymax))
    else:
      ax.set_ylim((0.,1.2*ymax))
  for i,pval in enumerate(pvals):
    add_pval(ax,pval[0],pval[1],pval[2],X,Y,.125*i,col=colors[pval[1]])
  ax.legend(loc='best',ncol=3,fontsize=14)
  if ylabel:
    if add_bars:
      ax.set_ylabel('occurances')
    if add_lines:
      ax.set_ylabel('empirical probability density')
  if xlabel:
    ax.set_xlabel(label)
  else:
    ax.set_xticklabels([])
  ax.set_yticks( np.asarray( ax.get_yticks(), dtype=np.int ) )
  ax.set_xticklabels(ax.get_xticks(),font)
  ax.set_yticklabels(ax.get_yticks(),font)
  return dict(fig=fig,ax=ax,colors=colors,lines=lines,dashes=dashes)

def hists(label,classes,data,#colors={},lines={},dashes={},
          savefig=False,fmts=['pdf'],lw=10.,figN=[1,2],
          do_bars=True,do_lines=True,title='',xlim=(-1.,.2),
          geoms={'line':'500x500+500+500','hist':'500x500+000+500'},pvals=[]):

  figs = {}

  fig = plt.figure(figN[0]); plt.clf()

  if tk is not None:
    tk.fig_geom(fig=fig,geom=geoms['line'])

  _ = hist(label,classes,data,fig=fig,xlim=xlim,
           #colors=colors,lines=lines,dashes=dashes,
           add_bars=False,add_lines=True,lw=lw,pvals=pvals)
  ax = _['ax']; colors = _['colors']; lines = _['lines']; dashes = _['dashes']

  ax.set_title(title)

  figs['line'] = fig

  fig = plt.figure(figN[1]); fig.clf()
  ax1 = plt.subplot(311); ax1.grid('on')
  ax2 = plt.subplot(312); ax2.grid('on')
  ax3 = plt.subplot(313); ax3.grid('on')

  if tk is not None:
    tk.fig_geom(fig=fig,geom=geoms['hist'])

  hist(label,[classes[0]],data,fig=fig,ax=ax1,xlim=xlim,
       colors=colors,lines=lines,dashes=dashes, 
       add_bars=True,add_lines=False,xlabel=False,ylabel=False)
  hist(label,[classes[1]],data,fig=fig,ax=ax2,xlim=xlim,
       colors=colors,lines=lines,dashes=dashes, 
       add_bars=True,add_lines=False,xlabel=False,ylabel=True)
  hist(label,[classes[2]],data,fig=fig,ax=ax3,xlim=xlim,
       colors=colors,lines=lines,dashes=dashes, 
       add_bars=True,add_lines=False,xlabel=True,ylabel=False)

  ax1.set_title(title)

  figs['hist'] = fig

  fis = []
  if savefig:
    for fmt in fmts:
      di = os.path.join('fig')
      if not os.path.exists(di):
        os.mkdir(di)
      di = os.path.join('fig',fmt)
      if not os.path.exists(di):
        os.mkdir(di)
      for fig in figs:
        fi = os.path.join(di,subj+'_'+tmt+'_'+base+'_'+fig+'.'+fmt)
        figs[fig].savefig(fi,pad_inches=0.,bbox_inches='tight',dpi=150)
        fis.append(fi)

  return figs,fis

def html():
  base = 'zero'
  #base = 'trial'

  Tmt = ['un','restricted']

  A = dict(); D = dict(); Subj = dict(); Sch = dict()

  savefig=True
  #savefig=False

  if savefig:
    html = '<html>\n <table>\n  <tr>\n'

  for tmt in Tmt:
    A[tmt],D[tmt],Subj[tmt],Sch[tmt] = load(tmt)

    fis = hists(A[tmt],Sch[tmt],tmt=tmt,savefig=savefig)
    if savefig:
      for fi in fis:
        html += '  <td><img src="'+fi+'"></img></td>\n'

    #for subj in sorted(Subj[tmt]):
    #  fis = hists(D[tmt][subj],Sch[tmt],subj=subj,tmt=tmt,savefig=True)
      #if savefig:
      #  for fi in fis:
      #    html += '  <td><img src="'+fi+'"></img></td>\n'

    if savefig:
      html += '  </tr>\n  <tr>\n'

  if savefig:
    html += '  </tr>\n </table>\n</html>'
    open('fig.html','w').write(html)
