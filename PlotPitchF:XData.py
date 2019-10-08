#!/usr/bin/env python
# coding: utf-8

# ## Wrangle Pitch F/x Data 

# In[7]:


import requests # apache http library
import xml.etree.ElementTree as ET # XML Parsing
import os # file and system level

import pandas as pd # pandas dataframes, structures
import numpy as np # used by pandas

#enable inline plotting
get_ipython().magic(u'matplotlib inline')

from termcolor import colored # colored output


# In[20]:


print ("Hello World")
print(colored("Hello World", "red"))


# # pull players into dictionary 

# In[8]:


url = "http://gd2.mlb.com/components/game/mlb/year_2014/month_06/day_18/gid_2014_06_18_colmlb_lanmlb_1/players.xml"
resp = requests.get(url)
print(colored(resp, "blue"))
xmlfile = "myplayers.xml"
with open(xmlfile, "wb") as f:
     f.write(resp.content)
statinfo = os.stat(xmlfile)
print(colored(xmlfile + " :" + str(round(statinfo.st_size/1024)) + "KB\n"))

# pull players <game><team><player>
tree = ET.parse(xmlfile)
game = tree.getroot()
teams = game.findall("./team")
playerDict = {}

for team in teams:
    print(team.attrib.get('name'))
    players = team.findall('player')
    for player in players:
        print("   ", player.attrib.get('id'), player.attrib.get('first'), player.attrib.get('last'))
        playerDict[player.attrib.get('id')] = player.attrib.get('first') + " " + player.attrib.get('last')


# In[9]:


playerDict["115629"]


# # Kershaw No-Hitter
# #Get Innings_All Data

# In[10]:


url = "http://gd2.mlb.com/components/game/mlb/year_2014/month_06/day_18/gid_2014_06_18_colmlb_lanmlb_1/inning/inning_all.xml"
resp = requests.get(url)
print(colored(resp,"blue"))
xmlfile = "mygame.xml"
with open(xmlfile, "wb") as f:
    f.write(resp.content)
statinfo = os.stat(xmlfile)
print(colored(xmlfile + ": " + str(round(statinfo.st_size/1024)) + " KB\n"))
    
tree = ET.parse(xmlfile)
root = tree.getroot()
print("Tree.root.tag: " + root.tag)

# unpack the game to get all the innings
for child in root:
    print(child.tag, child.attrib.get("num"))
    for frame in child:
        print("   ", frame.tag, frame.attrib)


# In[12]:


frames = ["top","bottom"]
pitchDictionary = {"FA": "fastball", "FF":"4-seam fb", "FT":"2-seam fb", "FC":"fb-cutter", "":"unknown", None:"none",
                  "FS": "fb-splitter", "SL":"slider", "CH":"changeup", "CU":"curveball","KC":"knuckle-curve",
                  "KN":"knuckleball","EP":"eephus","UN":"Unidentifief","PO":"pitchout", "SI":"sinker","SF":"split-finger"
                  }
totalPitchCount = 0 
innings = root.findall("./inning")
for inning in innings:
    for i in range(len(frames)):
        color = "blue" if i==0 else "yellow" #show top of inning and yellow for bottom
        print(colored("\nInning: " + inning.attrib.get("num") + "(" + frames[i] + ")", color, attrs = ['reverse']))
        fr = inning.find(frames[i])
        if fr is not None:
            for ab in fr.iter('atbat'):
                battername = playerDict[ab.get('batter')]
                abPitchCount = 0
                print(colored("   " + battername, color, attrs=['bold']))
                
                pitches = ab.findall("pitch")
                #print(pitches)
                for pitch in pitches:
                    abPitchCount += 1
                    totalPitchCount += 1
                    verbosePitch = pitchDictionary[pitch.get("pitch_type")]
                    print(colored("    " + str(abPitchCount) + ": " + verbosePitch, color))
                print("   " + colored(ab.attrib.get("event"),color, attrs=["underline"])) 
print("Total Pitches: " + str(totalPitchCount))


# # Load Inning detail into Dataframe
# # gameday pitch fields definitions: http://fastballs.wordpress.com/2007/08/02/glossary-of-the-gameday-pitch-fields/

# In[13]:


frames = ["top","bottom"]
pitchDF = pd.DataFrame(columns = ['pitchIdx', 'inning', 'frame', 'ab', 'abIdx', 'batter', 'stand', 'speed', 
                                 'pitchtype', 'px', 'pz', 'szTop', 'szBottom', 'des'])
totalPitchCount = 0 
topPitchCount = 0
bottomPitchCount = 0

innings = root.findall("./inning")
for inning in innings:
    for i in range(len(frames)):
        color = "blue" if i==0 else "yellow" #show top of inning and yellow for bottom
        print(colored("\nInning: " + inning.attrib.get("num") + "(" + frames[i] + ")", color, attrs = ['reverse']))
        fr = inning.find(frames[i])
        if fr is not None:
            for ab in fr.iter('atbat'):
                battername = playerDict[ab.get('batter')]
                standside = ab.get('stand')
                abIdx = ab.get('num')
                abPitchCount = 0
                print(colored("   " + battername, color, attrs=['bold']))
                
                pitches = ab.findall("pitch")
                #print(pitches)
                for pitch in pitches:
                    if pitch.attrib.get('start_speed') is None:
                       speed = 0
                    else:
                       speed = float(pitch.attrib.get('start_speed'))
                    pxFloat = 0.0 if pitch.attrib.get('px') == None else float('{0:.2f}'.format(float(pitch.attrib.get('px'))))
                    pzFloat = 0.0 if pitch.attrib.get('pz') == None else float('{0:.2f}'.format(float(pitch.attrib.get('pz'))))
                    szTop = 0.0 if pitch.attrib.get('sz_top') == None else float('{0:.2f}'.format(float(pitch.attrib.get('sz_top'))))
                    szBot = 0.0 if pitch.attrib.get('sz_bot') == None else float('{0:.2f}'.format(float(pitch.attrib.get('sz_bot'))))
                    print(pxFloat, pzFloat, szTop, szBot)
                    
                    abPitchCount += 1
                    totalPitchCount += 1
                    if frames[i]=='top':
                        topPitchCount+=1
                    else:
                        bottomPitchCount+=1
                    inn = inning.attrib.get("num")
                    des = pitch.get("des")
                    verbosePitch = pitchDictionary[pitch.get("pitch_type")]
                    print(colored("    " + str(abPitchCount) + ": " + verbosePitch, color))
                    pitchDF.loc[totalPitchCount] = [totalPitchCount, inn, frames[i], abIdx, abPitchCount, battername,
                                                   standside, speed, verbosePitch, pxFloat, pzFloat, szTop, szBot, des]
                print("   " + colored(ab.attrib.get("event"),color, attrs=["underline"]))  
print("Total Pitches: " + str(totalPitchCount))


# In[32]:


pitchDF.head(20)


# In[14]:


lines = pitchDF.plot.line(x='pitchIdx', y='speed', figsize=[12,6])


# In[15]:


pitchDF.info()
pitchDF['pitchIdx'] = pitchDF['pitchIdx'].astype('int')


# In[16]:


color = ['green' if f=='top' else 'yellow' for f in pitchDF['frame'].tolist()] 
pitchDF.plot(kind = 'scatter', x='pitchIdx', y='speed', color = color, figsize=[12,6])


# ### Select operations from the dataframe

# In[17]:


pitchDF[2:8] #non-inclusive range 


# In[18]:


pitchDF.loc[pitchDF['ab']=='12'] # match a column value


# In[19]:


pitchDF.loc[pitchDF['batter']=='Matt Kemp']


# In[20]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
get_ipython().magic(u'matplotlib inline')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')

plateWidthInFeet = 17 / 12 # plate is 17 inches
expandedPlateInFeet = 20 / 12 # adding 3 inch ball

szHeightInFeet = 3.5 - 1.5
ballInFeet = 3 / 12 
halfBallInFeet = ballInFeet / 2

# draw expanded zone
outrect = ax1.add_patch(patches.Rectangle((expandedPlateInFeet/-2,szBottom - halfBallInFeet), expandedPlateInFeet, szHeightInFeet + ballInFeet, color='lightblue'))
rect = ax1.add_patch(patches.Rectangle((plateWidthInFeet/-2, szBottom), plateWidthInFeet, szHeightInFeet))
plt.ylim(0,5)
plt.xlim(-2,2)




# ### Kershaw called balls and strikes (4 plots)

# In[21]:


pitchCalls = ['Called Strike', 'Ball']
sides = ['R','L']

colors = {'R':'red', 'L': 'black'}
markers = {'R': 'x','L':'o'}

for pitchCall in pitchCalls:
    for sidename in sides:
        df = pitchDF.loc[pitchDF['des']==pitchCall].loc[pitchDF['frame']=='top']
        mark = markers[sidename]
        colr = colors[sidename]
        #print(sidename)
        df_side = df.loc[pitchDF['stand']==sidename]
        ax1 = df_side.plot(kind='scatter',x='px',y='pz',marker = mark, color = colr, figsize=[8,8],ylim=[0,4],xlim=[-2,2])
        
        if sidename == 'R':
            xcoord = -1.6
            handers = "right handers"
        else:
            xcoord = 1.4
            handers = "left handers"
            
        txtbatter = ax1.text(xcoord, 2.5, sidename, style='italic', fontsize=24, color = colr)
        
        plateWidthInFeet = 17 / 12 # plate is 17 inches
        expandedPlateInFeet = 20 / 12 # adding 3 inch ball
        szBottom = df['szBottom'].iloc[0]
        szTop = df['szTop'].iloc[0]
        szHeightInFeet = szTop - szBottom
        ballInFeet = 3 / 12 
        halfBallInFeet = ballInFeet / 2

        # draw expanded zone
        outrect = ax1.add_patch(patches.Rectangle((expandedPlateInFeet/-2, szBottom - halfBallInFeet), 
                                                  expandedPlateInFeet, szHeightInFeet + ballInFeet, color='lightblue'))
        # draw formal zone
        rect = ax1.add_patch(patches.Rectangle((plateWidthInFeet/-2, szBottom), plateWidthInFeet, szHeightInFeet))
        outrect.zorder = -2
        rect.zorder = -1 
        
        ax1.set_xlabel(pitchCall + "s to " + handers + "(catcher's view).")
        ax1.set_ylabel('vertical location')
        ax1.set_aspect(aspect = 1)
                       
        plt.ylim(0,5)
        plt.xlim(-2.5,2.5)
        


# ### Dickerson AtBat #1

# In[22]:


df = pitchDF.loc[pitchDF['ab']=='1'] # first atbat of the game
    
ax1 = df.plot(kind='scatter',x='px',y='pz',color='red',figsize=[7,7],ylim=[0,4], xlim=[-2,2])
ax1.set_xlabel("Catcher's View - Horizontal location")
ax1.set_ylabel("Vertical Location")
ax1.set_aspect(aspect=1)  
    
plateWidthInFeet = 17 / 12 # plate is 17 inches
expandedPlateInFeet = 20 / 12 # adding 3 inch ball
szBottom = df['szBottom'].iloc[0]
szTop = df['szTop'].iloc[0]
szHeightInFeet = szTop - szBottom
ballInFeet = 3 / 12 
halfBallInFeet = ballInFeet / 2

# draw expanded zone
outrect = ax1.add_patch(patches.Rectangle((expandedPlateInFeet/-2, szBottom - halfBallInFeet), 
                                          expandedPlateInFeet, szHeightInFeet + ballInFeet, color='lightblue'))
# draw formal zone
rect = ax1.add_patch(patches.Rectangle((plateWidthInFeet/-2, szBottom), plateWidthInFeet, szHeightInFeet))
outrect.zorder = -2
rect.zorder = -1 
    
               
plt.ylim(0,5)
plt.xlim(-2.5,2.5)


# ### Label Pitches

# In[29]:


df=pitchDF.loc[pitchDF['ab']=='1']
#print(df)
stand = df['stand'].iloc[0]
batter = df['batter'].iloc[0]
inning = df['inning'].iloc[0]
frame = df['frame'].iloc[0]
suffix = "th"
suffix = "st" if inning == "1" else suffix
suffix = "nd" if inning == "2" else suffix
suffix = "rd" if inning == "3" else suffix


ax1 = df.plot(kind='scatter',x='px',y='pz',color='red',figsize=[7,7],ylim=[0,4], xlim=[-2,2])
ax1.set_xlabel("Catcher's View - " + batter + " (" + frame + " of the " + inning + suffix + ")")
ax1.set_ylabel("Vertical Location")
ax1.set_aspect(aspect=1)  
plt.ylim(0,5)
plt.xlim(-2.5,2.5)

plateWidthInFeet = 17 / 12 # plate is 17 inches
expandedPlateInFeet = 20 / 12 # adding 3 inch ball
szBottom = df['szBottom'].iloc[0]
szTop = df['szTop'].iloc[0]
szHeightInFeet = szTop - szBottom
ballInFeet = 3 / 12 
halfBallInFeet = ballInFeet / 2

# draw expanded zone
outrect = ax1.add_patch(patches.Rectangle((expandedPlateInFeet/-2, szBottom - halfBallInFeet), 
                                                  expandedPlateInFeet, szHeightInFeet + ballInFeet, color='lightblue'))
# draw formal zone
rect = ax1.add_patch(patches.Rectangle((plateWidthInFeet/-2, szBottom), plateWidthInFeet, szHeightInFeet))
outrect.zorder = -2
rect.zorder = -1 

if stand == 'R':
    standTextX = -1.5
else:
    standTextX = 1.4
tbatter = ax1.text(standTextX, 2.5, stand, style='italic',fontsize=24, 
                  bbox={'facecolor':'blue','alpha':0.2,'pad':10})

# label pitch points
ptypes = df["pitchtype"]
xcoords = df["px"]
ycoords = df["pz"]
speed = df["speed"]
abIdx = df["abIdx"]

for i, txt in enumerate(ptypes):
    print(abIdx.iloc[i], txt, xcoords.iloc[i], ycoords.iloc[i])
    txtDetail = str(abIdx.iloc[i]) + ") " + txt + " (" + str(int(speed.iloc[i])) + "mph)"
    txt = ax1.text(xcoords.iloc[i]+.1,ycoords.iloc[i]-.05,txtDetail,style='italic',fontsize=12)


# ### plot with legend

# In[40]:


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy.random import random

df=pitchDF.loc[pitchDF['ab']=='3']

pitchColors = {"FA":"b","FF":"b","FT":"y","FC":"k","":"w",None:"violet",
              "FS":"aqua","SL":"red","CH":"h","CU":"gold","KC":"tan",
              "KN":"^","EP":"*","UN":"X","PO":"s","SI":"_","SF":"seagreen"
              }
pitchMarkers = {"FA":"x","FF":"o","FT":"1","FC":"2","":"2",None:"4",
              "FS":"aqua","SL":"^","CH":"h","CU":"o","KC":"tan",
              "KN":"^","EP":"*","UN":"X","PO":"s","SI":"_","SF":"seagreen"
              }

#print(df)
stand = df['stand'].iloc[0]
batter = df['batter'].iloc[0]
inning = df['inning'].iloc[0]
frame = df['frame'].iloc[0]
suffix = "th"
suffix = "st" if inning == "1" else suffix
suffix = "nd" if inning == "2" else suffix
suffix = "rd" if inning == "3" else suffix

plt.figure(figsize=(7,7))

plateWidthInFeet = 17 / 12 # plate is 17 inches
expandedPlateInFeet = 20 / 12 # adding 3 inch ball
szBottom = df["szBottom"].iloc[0]
szTop = df["szTop"].iloc[0]
szHeightInFeet = szTop - szBottom
ballInFeet = 3 / 12 
halfBallInFeet = ballInFeet / 2

currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((expandedPlateInFeet/-2, szBottom - 0.125),expandedPlateInFeet, szHeightInFeet + 0.25, color="lightgrey"))
currentAxis.add_patch(Rectangle((plateWidthInFeet/-2, szBottom),plateWidthInFeet, szHeightInFeet, facecolor="lightgrey"))
if stand=='R':
    standTextX = -1.5
else:
    standTextX = 1.4
tbatter = currentAxis.text(standTextX,2.5,stand, style="italic", fontsize=24,
                          bbox={'facecolor':'blue','alpha':0.2, 'pad':10})

pitchList = []
availColors = ['b','r','y','k','w']
pitchesUsed = df['pitchtype'].unique()
abCount= len(df['speed'])
markersize = [0] * abCount
for i, sx in enumerate(df['speed'].tolist()):
    markersize[i] = int(sx - 70)*3
    thrownPitch = df['pitchtype'].tolist()[i]
    for j, (k,v) in enumerate(pitchDictionary.items()):
        if thrownPitch == v:
            pitchList.append(plt.scatter(df['px'].iloc[i],df['pz'].iloc[i],marker = pitchMarkers[k],
                            color = pitchColors[k], label = v, zorder=2))
            
print(markersize, colors, pitchesUsed)

plt.ylim(0,5)
plt.xlim(-2.5,2.5)
plt.gca().set_aspect('equal','box')

handles, labels= plt.gca().get_legend_handles_labels()
newLabels, newHandles = [],[]
for handle, label in zip(handles, labels):
    if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
plt.legend(newHandles, newLabels)

plt.xlabel("Catcher's view - " + batter + " (" + frame + "of the" + inning + suffix + ")")
plt.ylabel("vertical location")
plt.show()


# ### add ball strike count to the dataframe

# In[48]:


uniqDesList = pitchDF.des.unique()
#print(uniqDesList)
ballColList = []
strikeColList = []
ballCount = 0
strikeCount = 0
for index, row in pitchDF.iterrows():
    des = row['des']
    #print(des)
    if row["abIdx"] == 1:
        ballCount = 0
        strikeCount = 0
    
    # save count
    ballColList.append(ballCount)
    strikeColList.append(strikeCount)

    if "Ball" in des:
        ballCount += 1
    elif 'Foul' in des:
        if strikeCount is not 2:
            strikeCount += 1
    elif 'Strike' in des:
        strikeCount += 1
        
pitchDF['ballCount'] = ballColList
pitchDF['strikeCount'] = strikeColList
        
        


# In[49]:


pitchDF.head()


# ### kershaw tendency by pitchcount

# In[54]:


titleList = [] 
dataList = []
fig, axes = plt.subplots(4,3,figsize=(12,16))

for b in range(4):
    for s in range(3):
        #print("Count",str(b),str(s))
        df = pitchDF.loc[(pitchDF['ballCount']==b) & (pitchDF['strikeCount']==s) & (pitchDF['frame']=='top')]
        title = "count:" + str(b) + "-" + str(s) + " (" + str(len(df)) + ")"
        titleList.append(title)
        dataList.append(df)
        
for i, ax in enumerate(axes.flatten()):
    x = dataList[i].pitchtype.value_counts()
    l = dataList[i].pitchtype.unique()
    ax.pie(x, radius = 1, autopct = "%.1f%%", pctdistance = 0.9, labels = 1)
    ax.set_title(titleList[i])
plt.show()


# In[ ]:




