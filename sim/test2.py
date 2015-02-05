


import plotly.plotly as py
from plotly.graph_objs import *# auto sign-in with credentials or use py.sign_in()
import plotly.tools as tls
import time

#tls.set_credentials_file( \
#        username="borabanjanin", \
#        api_key="658g9odkf8")

tls.set_credentials_file(stream_ids=[ \
        "ohm659z4y0", \
        "i3yxmjesit", \
        "qx4x099wpc", \
        "e0i69ts2oz"])

stream_ids = tls.get_credentials_file()['stream_ids']

x1 = [1,2,3,5,6]
y1 = [1,4.5,7,24,38]

stream = Stream(
    token=stream_ids[1],  # (!) link stream id to 'token' key
    maxpoints=80      # (!) keep a max of 80 pts on screen
)

trace1 = Scatter(
    x=[0],
    y=[0],
    mode='lines+markers',
    stream=stream         # (!) embed stream id, 1 per trace
)

data = Data([trace1])
layout = Layout(title='Time Series')
fig = Figure(data=data,layout=layout)

unique_url = py.plot(fig, filename='test',auto_open=False)
print unique_url

s1 = py.Stream(stream_ids[1])

s1.open()
#time.sleep(5)
print data
try:
    for i in range(1,100):
        s1.write(dict(x=[i], y=[i]))
        #print dict(x=i, y=i)
        time.sleep(0.1)
except KeyboardInterrupt:
    s1.close

#for i in range(1,100):
#    py.plot(data, filename='s0_first_plot', auto_open=True)



#tls.set_credentials_file(stream_ids=[ \
#        "ohm659z4y0", \
#        "i3yxmjesit", \
#        "qx4x099wpc", \
#        "e0i69ts2oz"])
