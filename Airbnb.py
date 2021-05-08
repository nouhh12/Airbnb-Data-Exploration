import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from selenium import webdriver
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import time

df = pd.read_csv('AB_NYC_2019.csv')
for column in df.columns:
    if df[column].isnull().sum()>1:
        print(column,np.round(df[column].isnull().mean(),3),'% missing values')

#Fill missing reviews number with 0
df['reviews_per_month']=df['reviews_per_month'].fillna(0)
df.dropna(how='any',axis=1,inplace=True)
#df.duplicated().sum()==0 so no need to drop any duplicates

#Remove extreme values for better accuracy
norm=df.loc[((df['price']<175*1.5) & (df['price']>0)) & (df['room_type']=='Shared room') ]
#print(norm['price'].describe())
#print(norm['room_type'].shape)

#Opening chrome browser in icognito mode
""" options = webdriver.ChromeOptions()
options.add_argument("--incognito")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--ignore-ssl-errors")
driver=webdriver.Chrome(options=options,executable_path="D:\\Nouh\\Downloads\\chromedriver\\chromedriver.exe")
driver.get("https://www.google.com/")
search=driver.find_element_by_xpath("//*[@='gLFyf gsfi']")
search.send_keys("nyc")
element=driver.execute_script("arguments[0].value;",search)
print(element)
print("\nsearch:",search.text) """

attractions=pd.DataFrame({
    'Longitude':[-73.968285,-73.9857,-74.0445,-73.9855,-73.9969,-73.9787,-73.9772],
    'Latitude':[40.785091,40.7484,40.6892,40.7580,40.7061,40.7587,40.7527]
    })
attractions.index=['Central Park NYC',
                'Empire State Building',
                'Statue Of Liberty',
                'Times Square',
                'Brooklyn Bridge',
                'Rockefeller Center',
                'Grand Central Station']

""" for name in attractions.index:
    driver.find_element_by_id('searchterm').send_keys(name)
    btn=driver.find_element_by_xpath("/html/body/div[1]/div[5]/div[1]")
    driver.execute_script("arguments[0].click();", btn)
    driver.get("https://www.gps-latitude-longitude.com/")
    time.sleep(1)
    nameof=driver.find_element_by_id("searchterm").text
    longitude=driver.find_element_by_id("lng").text
    latitude=driver.find_element_by_id("lat").text
    print(nameof,"longs:",longitude,"  lats:",latitude,"  \n",attractions.loc[name]) """
#Took absolute of coordinates to calculate distance rather than direction
x=abs(norm['longitude']-attractions['Longitude'].mean())
y=abs(norm['latitude']-attractions['Latitude'].mean())
proximity=x+y

X =np.array(proximity).reshape(-1,1)
y=np.array(norm['price']).reshape(-1,1)
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.1, random_state= 0)
regr=LinearRegression()
regr.fit(X_train,y_train)
print(regr.score(X_test,y_test))
y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
#Boundingbox=[df.longitude.min(),df.longitude.max(),df.latitude.min(),df.latitude.max()]
#nycmap=plt.imread('Airbnb/map.png')
fig, ax = plt.subplots(figsize = (15,6))
#ax.scatter(x, y, zorder=1, alpha= 0.2, s=0.2)
ax.set_title('Plotting Host Locations NYC Map')
#ax.set_xlim(Boundingbox[0],Boundingbox[1])
#ax.set_ylim(Boundingbox[2],Boundingbox[3])
#ax.imshow(nycmap, zorder=0, extent = Boundingbox, aspect= 'equal')
#sns.scatterplot(x=x,y=y,hue=df['neighbourhood_group'])

#bar chart shows prices are highest in manhattan and lowest in the bronx
#plt.figure(figsize=(15,6))
#sns.barplot(df['neighbourhood_group'],df['price'],hue=df['room_type'],ci=None)
sns.scatterplot(y=proximity,x=norm['price'])

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
plt.grid(True)
plt.show()
