import matplotlib.pylab as plt
from matplotlib import rcParams
import numpy as np
import csv
import six,sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')

# https://www.youtube.com/watch?v=UHBmv7qCey4
# https://www.youtube.com/watch?v=gmok1h8wG-Q

    #### Utility ###########

def ReadFile_getData(fileName): # formatas: [data<...>, boolean], paskutinis turetu buti reikšminis bool
    with open(fileName) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        entitiyList = []
        for row in csvReader:
            data = []
            for col in range(len(row)-1):
                data.append(float(row[col].strip()))
            if row[-1].strip().lower() in ['1','-1','true','t','yes','y']:
                indicator = 1
            else: indicator = 0
            entity = (tuple(data), indicator)
            entitiyList.append(entity)
    return entitiyList

def ReadFile_getRules(fileName): # formatas: [data<...>, boolean], paskutinis turetu buti reikšminis bool
    with open(fileName) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        rulesList = []
        for row in csvReader:
            rulesList.append(row[0].strip())
    return rulesList

def DrawGraphic(dataList, rules, toPNG=False, discription='signalas'): # tikimasi dataList = [[x,y,bool], ...]
    #plt.xlabel('xlabel', fontsize=18)
    #window.tight_layout()
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure()
    ax = plt.axes()

    clr = "black"
    for entity in dataList:
        if (entity[1] == 0): clr = "red"
        else: clr = "green"
        plt.scatter(entity[0][0], entity[0][1], 70, color=clr) # x, y and bool
    

    xMin, xMax = 0, 6 #ax.get_xlim()
    yMin, yMax = 0, 6 #ax.get_ylim()
    for rule in rules:
        if (rule[1] in ['>','>=']): clr = "green"
        else: clr = "red"
        if (rule[0] == 0):
            plt.plot([rule[2], rule[2]], [yMin, yMax], 2, color=clr) # x, y and bool
        elif (rule[0] == 1):
            plt.plot([xMin, xMax], [rule[2], rule[2]], 2, color=clr) # x, y and bool


    # Draw or Imaging
    if (toPNG == True):
        plt.savefig(discription +".png") #issaugo "figura"(plot) faile
    else:
        plt.show ()


#### AdaBoost class ########################

class AdaBoost:
    def __init__(self, train_data, classifiers, maxIter=1000):
        self.Hx = None
        self.maxIter = maxIter
        self.learning_Iter = 0
        self.learned = False
        self.train_data = train_data
        self.N = len(self.train_data)

        self.weights = []
        for i in range(self.N): self.weights.append(1.0/self.N)
        
        self.classifiers = classifiers
        self.Rules = []
        self.VotingPowers = []

        while(self.learned == False):
            self.learn()
            self.learning_Iter += 1
            #print("New Weights: ", self.weights)
            
    
    def getErrorRates(self):
        errorRates = []
        for classif in self.classifiers:
            errRate = 0.0
            for i in range(self.N):
                data = self.train_data[i][0]
                dataSign = self.train_data[i][1]
                if(dataSign != classif(data)):
                    errRate += self.weights[i] # errorRate(weightSum)    
            errorRates.append(errRate)
        return errorRates

    def learn(self):
        # Calculate errorRate of classifier
        errorRates = self.getErrorRates()
        #print("Error Rates: ", errorRates)
        smallestErrIndx = errorRates.index(min(errorRates))
        smallest_ErrorRate = errorRates[smallestErrIndx]
        if (smallest_ErrorRate != 0.0 ):
            votingPower = 0.5 * np.log((1.0 - smallest_ErrorRate) / smallest_ErrorRate)  # daug kur zymima alfa
        else: votingPower = 0.5 * 1
        # Gaminam H(x) = sgn(vp*h(x)+...) polinoma
        self.Rules.append(self.classifiers[smallestErrIndx])
        self.VotingPowers.append(votingPower)
        print ('Smallest ErrorRate = %.2f & VotingPower (alfa) = %.2f'%(smallest_ErrorRate, votingPower))

        # Check if finished:
            # H(x) good enough - Human Boosting (kompiuteriu per brangu tikrinti)
        self.learned = self.doneLearning()
            # Enough rounds (saugiklis)
            # Nebeliko geru klasifikatoriu (Best errorRate >= 0.5)
        if (self.maxIter <= self.learning_Iter or smallest_ErrorRate >= 0.5 or self.learned):
            self.learned = True
            return

        # recalc weights:      
        self.recalc_Weights(smallest_ErrorRate)
    
    def recalc_Weights(self, smallest_ErrorRate):
        # kurie taskai, kurie nepateko i geriausio klasifikatorio klaidas - right, else -wrong
        for i in range(self.N):
            point = self.train_data[i][0]
            dataSign = self.train_data[i][1]
            rule = self.Rules[-1]
            if(dataSign != rule(point)):
                self.weights[i] = 0.5 / smallest_ErrorRate * self.weights[i]
            else:
                self.weights[i] = 0.5 / (1.0 - smallest_ErrorRate) * self.weights[i]

    def get_Hx(self, data, dataSign):
        Hx_List = []
        if (dataSign == False): dataSign = -1
        for i in range(len(self.Rules)):
            ruleRslt = self.Rules[i](data)
            if (ruleRslt == False): ruleRslt = -1
            Hx_List.append( self.VotingPowers[i] * ruleRslt )

        Hx = np.sign(sum(Hx_List))
        return Hx

        if (dataSign == 0): print("KEBABABS", sum(Hx_List))
        print ( data, np.sign(dataSign) == Hx)



    def doneLearning(self):
        Hx_List = []
        for (data, dataSign) in self.train_data:
            Hx = self.get_Hx(data, dataSign)
            if (dataSign == False): dataSign = -1
            if(np.sign(dataSign) != Hx):
                return False # go learn more...
                # Arba galima padaryti - kiek procentu gaunama true, pvz 95%
            Hx_List.append(Hx)
        # Alg. became perfect!
        i = 0
        for (data, dataSign) in self.train_data:
            if (dataSign == False): dataSign = -1
            print ( data, np.sign(dataSign) == Hx_List[i])
            i+=1
        return True
        
            
######## Main() ###############################################
'''
# x & y, true of false
data = [
    ((1, 5),  1), # A
    ((5, 5),  1), # B
    ((3, 3),  0), # C
    ((1, 1),  1), # D
    ((5, 1),  1), # E
    ((3.2, 2.8),  0), # F
    ((2.7, 3.4),  0)  # G
]

classifiers = [
    lambda data: data[0] < 2,
    lambda data: data[0] < 4,
    lambda data: data[0] < 6,
    lambda data: data[0] > 2,
    lambda data: data[0] > 4,
    lambda data: data[0] > 6,
    lambda data: data[1] < 4,
    lambda data: data[1] > 2
]
'''

data = ReadFile_getData('data.csv')
rules = ReadFile_getRules('rules.csv')

# clasification funcions (weak learners)
classifiers = []
for i in rules:
    classifiers.append(eval(i))

booster = AdaBoost(data, classifiers)


########### Drawing ################
#(dimension, sign, rule)
ruleList = [
    (0,'<', 2),
    (0,'<', 4),
    (0,'<', 6),
    (0,'>', 2.2),
    (0,'>', 3.5),
    (0,'>', 5.5),
    (1,'<', 4),
    (1,'>', 2),
]

DrawGraphic(data, ruleList, False, 'data')

    
