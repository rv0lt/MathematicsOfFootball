#The basics
import pandas as pd
import numpy as np
import json

#Plotting
import matplotlib.pyplot as plt
import FCPython 

#Statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf

###declare auxilary functions

#filter all the events done by a set of players given
def filterEventsByPlayers(eventsDf,playersDF):
    eventsDefenders = pd.DataFrame()
    for i,player in playersDF.iterrows():
        playerId = player['wyId']
        aux = eventsDf[eventsDf["playerId"] == playerId]
        eventsDefenders = eventsDefenders.append(aux)
    return eventsDefenders


def createChainId(eventsDf):
    """
    create the chainID

    chanID starts with 1, store teamId value

    if teamId doesnt change, chainId is the same value
    if teamID changes, check next teamID
        if return to previous teamID insert negative chanID
        if teamID is still diferent, increase chainID for this two events and increase chanId value
    """
    eventsDfWithChain = eventsDf

    chainId=1
    teamId = eventsDf.head(1)["teamId"].values[0]
    eventsDfWithChain["chainId"] = 0
    eventsDfWithChain["xg"] = None

    eventsDf['start'] = eventsDf.index
    eventsDf.start = eventsDf.start.shift(1, fill_value=eventsDf.index.min())

    #when using iterrows we shouldnt modify the object, so we create a copy

    for i,row in eventsDf.iterrows():
        aux = eventsDf.loc[row.start : i, ['teamId']]

        if (row["teamId"] == teamId):
            eventsDfWithChain.at[i,"chainId"] = chainId
        else:
            nextTeamId = aux["teamId"].values[1]
            if (nextTeamId == teamId):
                eventsDfWithChain.at[i,"chainId"] = -chainId
            else:
                chainId +=1
                teamId = nextTeamId
                eventsDfWithChain.at[i,"chainId"] = chainId
    return eventsDfWithChain


def createShotsModel(eventsDfWithChain):
    #Create a data set of shots.
    trainDf = pd.DataFrame(eventsDfWithChain)
    pd.unique(trainDf['subEventName'])
    shotsDf=trainDf[trainDf['subEventName']=='Shot']
    shotsModel=pd.DataFrame(columns=['Goal','X','Y'])

    #Go through the dataframe and calculate X, Y co-ordinates.
    #Distance from a line in the centre
    #Shot angle.
    #Details of tags can be found here: https://apidocs.wyscout.com/matches-wyid-events
    for i,shot in shotsDf.iterrows():
        
        header=0
        for shottags in shot['tags']:
            if shottags['id']==403:
                header=1
        #Only include non-headers        
        if not(header):        
            shotsModel.at[i,'X']=100-shot['positions'][0]['x']
            shotsModel.at[i,'Y']=shot['positions'][0]['y']
            shotsModel.at[i,'C']=abs(shot['positions'][0]['y']-50)
        
            #Distance in metres and shot angle in radians.
            x=shotsModel.at[i,'X']*105/100
            y=shotsModel.at[i,'C']*65/100
            shotsModel.at[i,'Distance']=np.sqrt(x**2 + y**2)
            a = np.arctan(7.32 *x /(x**2 + y**2 - (7.32/2)**2))
            if a<0:
                a=np.pi+a
            shotsModel.at[i,'Angle'] =a
        
            #Was it a goal
            shotsModel.at[i,'Goal']=0
            for shottags in shot['tags']:
                    #Tags contain that its a goal
                    if shottags['id']==101:
                        shotsModel.at[i,'Goal']=1
    return shotsModel

def calculate_xG(sh):    
   bsum=b[0]
   for i,v in enumerate(modelVariables):
       bsum=bsum+b[i+1]*sh[v]
   xG = 1/(1+np.exp(bsum)) 
   return xG   

#add all the xG values asociated to each goal to their respectives chains
def addXg(goals,eventsDfWithChain):
    for i,goal in goals.iterrows():
        xg = goal["xG"]
        chainId = eventsDfWithChain.at[i,"chainId"]
        chainEvents = eventsDfWithChain[eventsDfWithChain["chainId"] == chainId]
        for j,event in chainEvents.iterrows():
            eventsDfWithChain.at[j,"xg"] = xg
    return eventsDfWithChain

#filter all the events given and return only the interceptions
def filterInterceptionsEvents(eventsDefenders):
    interceptionsDf = pd.DataFrame()

    for i,event in eventsDefenders.iterrows():
        for tag in event['tags']:
            if tag['id'] == 1401:
                interceptionsDf = interceptionsDf.append(event)
    return interceptionsDf

#return a new Dataframe with the xg chain and interceptions metrics
def getMetrics(eventsDFWithChain,interceptionsDf):
    aux = eventsDFWithChain.groupby("playerId")["xg"].sum()
    metricsDf = aux.to_frame()

    metricsDf["interceptions"] = 0
    interceptionsDf["self"] = 1
    aux = interceptionsDf.groupby("playerId")["self"].count()
    for i,player in aux.to_frame().iterrows():
        metricsDf.at[i,"interceptions"] = player["self"]
    metricsDf = metricsDf[metricsDf["interceptions"] > 0]
    return metricsDf

#return a list of the best players based on the metrics and limits given
def generateListOfPlayers(metricsDf,xgLimit,interceptionsLimit,numberPlayers):
    aux = metricsDf[(metricsDf["xg"] > xgLimit) & (metricsDf["interceptions"] > interceptionsLimit)]
    listOfPlayers = aux.copy()
    listOfPlayers["firstName"] = None
    listOfPlayers["middleName"] = None
    listOfPlayers["lastName"] = None
    listOfPlayers["shortName"] = None

    for i,player in aux.iterrows():
        player = playersDf[playersDf["wyId"] == i]
        listOfPlayers.at[i,"firstName"] = player["firstName"].values[0]
        listOfPlayers.at[i,"middleName"] = player["middleName"].values[0]
        listOfPlayers.at[i,"lastName"] = player["lastName"].values[0]
        listOfPlayers.at[i,"shortName"] = player["shortName"].values[0]
    return listOfPlayers.head(numberPlayers)
###
print("--------------------------------------------------\n")
print("-------------------------\n ASIGMENT 2 SCRIPT\n-------------------------\n")
print("---------------------------------------\n ESTIMATED EXECUTION TIME IS: 15-20 MINUTES\n---------------------------------------\n")


#Load Data
print("-------------------------\n LOADING DATA\n-------------------------\n")
with open('Wyscout/events/events_England.json') as f:
    eventsData = json.load(f)
eventsDf = pd.DataFrame(eventsData)

with open('Wyscout/players.json') as f:
    playersData = json.load(f)
playersDf = pd.DataFrame(playersData)

print("-------------------------\n DATA LOADED\n-------------------------\n")
##

'''
we now filter by defenders, in the players dataframe, we have a json in one of the columns indicating
the different names of the players position, so we denormalize that column and then filter to get only the defenders
'''

#https://stackoverflow.com/questions/38231591/split-explode-a-column-of-dictionaries-into-separate-columns-with-pandas/63311361#63311361

# normalize the column of dictionaries and join it to df
playersDf = playersDf.join(pd.json_normalize(playersDf.role))

# drop role column (optional)
#aux.drop(columns=['role'], inplace=True)

#Filter by player's position
playersDf = playersDf[playersDf['code2'] == 'DF']

#get all the events done by defenders
eventsDefenders = filterEventsByPlayers(eventsDf,playersDf)

#create a new dataframe with a new column refering to the chainId
print("-------------------------\n CREATING CHAIN\n-------------------------\n")
eventsDfWithChain = createChainId(eventsDf)

###########
'''
now we can create and fit a model for the xg value
when done, we search all the goals within the chain, and add the xg to their chains
'''
print("-------------------------\n CREATING SHOTS MODEL\n-------------------------\n")
shotsModel = createShotsModel(eventsDfWithChain)

# List the model variables you want here
#model_variables = ['Angle','Distance','X','C']
modelVariables = ['Angle','Distance']
model=''
for v in modelVariables[:-1]:
    model = model  + v + ' + '
model = model + modelVariables[-1]

#Fit the model
test_model = smf.glm(formula="Goal ~ " + model, data=shotsModel, 
                           family=sm.families.Binomial()).fit()
print(test_model.summary())        
b=test_model.params

print("-------------------------\n CLACULATING XG\n-------------------------\n")
#Add an xG to my dataframe
xG=shotsModel.apply(calculate_xG, axis=1) 
shotsModel = shotsModel.assign(xG=xG)

print("-------------------------\n INSERTING XG INTO THE CHAIN\n-------------------------\n")
goals = shotsModel[shotsModel["Goal"] == 1]
#add all the xG values asociated to each goal to their respectives chains
eventsDfWithChain = addXg(goals, eventsDfWithChain)
eventsDfWithChain = eventsDfWithChain[["eventId", "playerId", "matchId", "teamId", "chainId", "xg"]]
eventsDFWithChain = eventsDfWithChain[eventsDfWithChain["xg"] > 0] 

'''
now we will take the number of interceptions for each player and add it on the dataframe
'''
print("-------------------------\n GETTING INTERCEPTIONS\n-------------------------\n")
interceptionsDf = filterInterceptionsEvents(eventsDefenders)

print("-------------------------\n CALCULATING METRICS\n-------------------------\n")
metricsDf = getMetrics(eventsDFWithChain,interceptionsDf)

#create plots
print("-------------------------\n CREATING PLOT\n-------------------------\n")
ax = metricsDf.plot.scatter(x="xg", y="interceptions")
ax.axvline(x=1.5,ymin=0, ymax=1, color="red")
ax.axhline(y=125, xmin=0, xmax=1, color="red")
plt.show()

listOfPlayers = generateListOfPlayers(metricsDf,1.5,125,10)
print("-------------------------\n LIST OF THE 10 BEST PLAYERS\n-------------------------\n")
print(listOfPlayers)
print("-------------------------\n END OF THE PROGRAM\n-------------------------\n")
print("--------------------------------------------------\n")
