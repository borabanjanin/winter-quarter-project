import modelplot as modelplot
import modelwrapper
import puck as puck
import time


if __name__ == "__main__":
    mw = modelwrapper.ModelWrapper()
    mc = modelwrapper.ModelConfiguration(mw)
    template = mc.jsonLoadTemplate('template')
    template = mw.packConfiguration(template)
    ID = mw.runTrial("puck.Puck",mc.jsonSaveConfiguration(mc.generateConfiguration(template)),["t","theta","x","y",])
    #ID = mw.runTrial("lls.LLS",mc.jsonSaveConfiguration(mc.generateConfiguration(template)),["t","theta","x","y","fx","fy","v","delta","omega"])
