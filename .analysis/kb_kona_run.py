import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Read the data
df = pd.read_csv('kristian_blummenfelt_copyright_entalpi_as.csv')

#convert datetime column to datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# create an new column with the time in seconds
df['time_in_seconds'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds()

#drop rows with 'time_in_seconds > 9656
df = df[df['time_in_seconds'] < 9675]

#replace NaN values with mean
df['stride_length'] = df['stride_length'].fillna(df['stride_length'].mean())

#calculate correlations for the dataframe
print(df.corr())

#function to calculate the ratio between heartrate and speed, cadence and stride length. This is used to scale the trendlines.
def calc_rat(clmn_1,clmn_2):
    rat = clmn_1.mean() / clmn_2.mean()
    return rat

# Plot the data
df.plot.scatter(x='time_in_seconds', y='heartrate',c='core_temperature', colormap='viridis')
# plot trendline for speed
z = np.polyfit(df['time_in_seconds'], df['speed']*calc_rat(df['heartrate'],df['speed']), 2)
p = np.poly1d(z)
plt.plot(df['time_in_seconds'],p(df['time_in_seconds']),"r--",color='red')
#plot trendline for cadence
z = np.polyfit(df['time_in_seconds'], df['cadence']*calc_rat(df['heartrate'],df['cadence']), 2)
p = np.poly1d(z)
plt.plot(df['time_in_seconds'],p(df['time_in_seconds']),"r--",color='blue')
#plot trendline for stride_length
z = np.polyfit(df['time_in_seconds'], df['stride_length']*calc_rat(df['heartrate'],df['stride_length']), 2)
p = np.poly1d(z)
plt.plot(df['time_in_seconds'],p(df['time_in_seconds']),"r--",color='green')
# draw vertical line at 6240 seconds (Gustav breakes away)
plt.axvline(x=6580, color='black',linestyle = '--',alpha=0.5)
# add text to the vertical line
plt.text(6587, 152, 'Gustav breaks away',color='black',alpha =0.5,rotation=90)
#legend
plt.legend(['Heartrate','Speed','Cadence','Stride Length'])
#plot elevation on a second y axes as bar chart
ax2 = plt.twinx()
ax2.bar(df['time_in_seconds'], df['elevation'], color='grey', alpha=0.2)
#legend
plt.legend(['Elevation'])
#label the axes
plt.xlabel('Time in seconds')
plt.ylabel('Heartrate')
ax2.set_ylabel('Elevation')
#Title
plt.title('Kristian Blummenfelt Kona Run 2022 (Temp: 29 C, Hum: 68%, Start Time: 11:27am)')
#show the plot
plt.show()

# Plot the data on a map using plotly express
fig = px.scatter_mapbox(df, 
                        lat="latitude", 
                        lon="longitude", 
                        hover_name="time_in_seconds", 
                        hover_data=["speed", "cadence", "stride_length", "heartrate", "elevation", "core_temperature"],
                        color="core_temperature",
                        color_continuous_scale='viridis',
                        zoom=11, 
                        height=800,
                        width=800)

fig.update_layout(mapbox_style="open-street-map")
fig.show()








