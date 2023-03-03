import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df_kb = pd.read_csv('kristian_blummenfelt_copyright_entalpi_as.csv')
df_gi = pd.read_csv('gustav_iden_copyright_entalpi_as.csv')

#prepend all column names except od datetime with 'kb_' or 'gi_'
df_kb.columns = ['kb_' + str(col) if col != 'datetime' else col for col in df_kb.columns]
df_gi.columns = ['gi_' + str(col) if col != 'datetime' else col for col in df_gi.columns]

#merge dataframe on datetime column
df = pd.merge(df_kb, df_gi, on='datetime')

#convert datetime column to datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# create an new column with the time in seconds
df['time_in_seconds'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds()

#drop rows with 'time_in_seconds > 9675
df = df[df['time_in_seconds'] < 9675]

# calculate difference between kb_speed and gi_speed
df['speed_diff'] = df['kb_speed'] - df['gi_speed']

# calculate difference between kb_stride_length and gi_stride_lenth
df['stride_diff'] = df['kb_stride_length'] - df['gi_stride_length']

# calculate difference between kb_cadence and gi_cadence
df['cadence_diff'] = df['kb_cadence'] - df['gi_cadence']

#scale the cadence difference to the same scale as speed and stride length
df['cadence_diff'] = df['cadence_diff'] / 7

# Plot stride length difference and speed diffrence on a bar plot
df.plot.bar(x='time_in_seconds', y=['stride_diff','speed_diff','cadence_diff'], color=['orange','blue','brown'],width=0.7)
# limit y axes to -1 and 1.5
plt.ylim(-1.2, 1.2)
# reduce labels on x axis
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], ['0', '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000'])
# draw vertical line at 6120 seconds
plt.axvline(x=6420, color='green',linestyle='--')
# add text to the vertical line
plt.text(6427, 0.5, 'Gustav Breakaway', color='green', rotation=90)
# plot exponential moving average (10min) of speed difference
plt.plot(df['time_in_seconds'], df['speed_diff'].ewm(span=600).mean(), color='navy',alpha=0.5)
# set labels
plt.xlabel('Time in Seconds')
plt.ylabel('Speed Difference (m/s)')
# set legend
plt.legend(['GI Breakaway','Speed EMA(600s)','Stride Length Difference','Speed Difference','Cadence Difference'], loc='upper left')
#plot elevation on a second y axes as bar chart
ax2 = plt.twinx()
ax2.bar(df['time_in_seconds'], df['kb_elevation'], color='grey', alpha=0.2)
# align 0.0 on y axes
plt.ylim(-30, 30)
# set labels
plt.ylabel('Elevation (m)')
# set title
plt.title('Kristian Blummenfelt vs Gustav Iden Kona Run 2022 (Temp: 29 C, Hum: 68%, Start Time: 11:27am)')
# show plot
plt.show()






