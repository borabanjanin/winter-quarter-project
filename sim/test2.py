class ModelOptimize(object):

    #Penalty for missing a data point per index
    MISSING_PENALTY = 100

    columnIDs = {
        0:"CartX",
        1:"CartVelocity",
        2:"CartAcceleration",
        3:"TarsusCart1_x",
        4:"TarsusCart2_x",
        5:"TarsusCart3_x",
        7:"TarsusCart5_x",
        6:"TarsusCart4_x",
        8:"TarsusCart6_x",
        9:"TarsusCart1_y",
        10:"TarsusCart2_y",
        11:"TarsusCart3_y",
        12:"TarsusCart4_y",
        13:"TarsusCart5_y",
        14:"TarsusCart6_y",
        15:"TarsusBody1_x",
        16:"TarsusBody2_x",
        17:"TarsusBody3_x",
        18:"TarsusBody4_x",
        19:"TarsusBody5_x",
        20:"TarsusBody6_x",
        21:"TarsusBody1_y",
        22:"TarsusBody2_y",
        23:"TarsusBody3_y",
        24:"TarsusBody4_y",
        25:"TarsusBody5_y",
        26:"TarsusBody6_y",
        27:"TarsusCombined_x",
        28:"TarsusCombined_dx",
        29:"C0",
        30:"C1",
        31:"TarsusBody1_vx",
        32:"TarsusBody2_vx",
        33:"TarsusBody3_vx",
        34:"TarsusBody4_vx",
        35:"TarsusBody5_vx",
        36:"TarsusBody6_vx",
        37:"TarsusBody1_vy",
        38:"TarsusBody2_vy",
        39:"TarsusBody3_vy",
        40:"TarsusBody4_vy",
        41:"TarsusBody5_vy",
        42:"TarsusBody6_vy",
        43:"Roach_x",
        44:"Roach_y",
        45:"Roach_vx",
        46:"Roach_vy",
        47:"Roach_pitch",
        48:"Roach_roll",
        49:"Roach_yaw",
        50:"Roach_theta",
        51:"Roach_dtheta",
        52:"Roach_v",
        53:"Roach_heading",
        54:"Roach_omega",
        55:"Roach_phaser_phase",
        56:"Roach_phaser_residual",
        57:"Roach_xv_phase",
        58:"Roach_xv_residual",
    }

    label = {
        'x':'Roach_x',
        'y':'Roach_y',
        'v':'Roach_v',
        'theta':'Roach_theta',
        'omega':'Roach_dtheta'}

    parmList = ["t","theta","x","y","fx","fy","v","delta","omega"]

    #TO DO: move to functions
    indexValues = {'begin':0,'end':200,'offset':283}

    varOpt = ['x','y','v','theta','omega']

    varCost=['x','y']

    def __init__(self, modelwrapper):
        self.modelwrapper = modelwrapper
        self.mc = ModelConfiguration(modelwrapper)
        #ms = mp.ModelStreamer()
        #self.stream1, self.stream2 = ms.createConnection()
        self.optimizationData = {}

    def displayBestTrial(self):
        best = pd.Series(self.modelwrapper.trials['Cost'])
        best.reset_index()
        bestIndex = (best[best == min(best)].index)[0]
        bestOB = self.modelwrapper.observations[bestIndex]
        print 'Best trial: ' + str(bestIndex)
        self.stream1.write(dict(x=list(bestOB['x']), y=list(bestOB['y'])))

    def runTemplate(self,templateName):
        template = self.mc.jsonLoadTemplate(templateName)
        template = self.modelwrapper.packConfiguration(template)
        x0 = self.initialGuess(template)
        #ID = self.modelwrapper.runTrial("lls.LLS",self.mc.jsonSaveConfiguration(self.mc.generateConfiguration(template)),ModelOptimize.parmList)
        #return x0, ID
        return 1,1


    def checkModelIndex(observation, i):
        if any(observation.index == i):
            return True
        else:
            return False

    '''
    def checkDataIndex(self, i, variables):
        for variables in variables:
            if isnull(data.ix[i,variable]):
                raise Exception("SimulatedAnealing: Null data " + str(i) + variable)
            else:
                return True
    '''

    def cost(x0):
        summation = 0.0
        observation = self.modelwrapper.observations[self.modelwrapper.trialID()-1]
        for i in range(indexValues['begin'],indexValues['end']+1):
            if self.checkModelIndex(observation,i):
                currentDiff = []
                for var in ModelOptimize.varCost:
                    dataName = ModelOptimize.label[var]
                    summation += (float(self.optimizationData.ix[i + indexValues['offset'], dataName]) - float(observation.ix[i, var]))**2
            else:
                summation += MISSING_PENALTY
        return summation

    #The minimizer just returns values
    def noChange(fun, x0, args, **options):
        return spo.OptimizeResult(x=x0, fun=fun(x0), success=True, nfev=1)

    #Used to stream data to plotly
    def streamTrial(ID):
        observation = self.modelwrapper.observations[ID]
        k=[]
        j=[]
        h=[]
        l=[]
        for i in range(indexValues['begin'],indexValues['end']+1):
            if self.checkModelIndex(observation,i):
                    k.append(observation.ix[i, 'x'])
                    j.append(observation.ix[i, 'y'])
                    h.append(self.optimizationData.ix[i + indexValues['offset'], self.label['x']])
                    l.append(self.optimizationData.ix[i + indexValues['offset'], self.label['y']])

        self.stream1.write(dict(x=k, y=j))
        self.stream2.write(dict(x=h, y=l))

    #Used in basinhopping callback
    #ModelWrapper hook
    def modelSimulate(x, f, record):
        print f

        variables = {}
        for i in range(x.size):
            variables[ModelOptimize.varOpt[i]] = x[i]
        confID = self.mc.generateConfiguration(variables)
        confFile = self.mc.jsonSaveConfiguration(confID)
        ID = mw.runTrial("lls.LLS",confFile,ModelOptimize.parmList)
        if ID % 20 == 0:
            self.streamTrial(ID)
        self.modelwrapper.updateCost(ID-1, f)
        self.modelwrapper.updateDataID(ID,DATA_ID)
        self.modelwrapper.updateCost(ID, 1000000.00)

    def initialGuess(self,template):
        x0 = []
        for var in ModelOptimize.varOpt:
            x0.append(template[var])
        return x0

    def optimizationLoop(self, x0):
        minimizer_kwargs = {"method":self.noChange, "jac":False}
        try:
            #ret = spo.basinhopping(self.cost,x0,minimizer_kwargs=minimizer_kwargs, \
            #niter=100,callback=self.modelSimulate,T=10.,stepsize=8.,interval=60)
            #print ret
            print 'happy'
        except:
            print "exit optimization"

    def runOptimize(self, modelName, template, dataID):
        print 'happy'
        x0, ID = self.runTemplate(template)
        #dataID = self.modelwrapper.csvLoadData(dataID)
        #self.optimizationData = self.modelwrapper.data[dataID]
        #self.optimizationLoop(x0)
        #self.modelwrapper.csvReleaseData(dataID)
        #print self.modelwrapper.observations

    def __del__(self):
        #print len(gc.get_referrers(self.modelwrapper.observations[0]))
        print "opt"

if __name__ == "__main__":
    print
