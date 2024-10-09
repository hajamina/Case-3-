#!/usr/bin/env python
# coding: utf-8

# ## Data importeren/inspecteren

# In[385]:


import pandas as pd
airports_extended = pd.read_csv('airports-extended-clean.csv', delimiter = ';')
airports_extended.rename(columns={'Name': 'Airport'},inplace =True)

airports_extended.head(50)


# In[386]:


print(pd.isna(airports_extended).sum()) # aantal missende waarden per kolom  


# In[387]:


filtered_dataset = airports_extended[airports_extended['City'].isnull()]


# In[388]:


city_country_mapping = airports_extended.groupby('Country')['City'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
#missende waarden in kolomen invullen 
airports_extended['City'] = airports_extended.apply(lambda row: city_country_mapping[row['Country']] if pd.isnull(row['City']) else row['City'], axis=1)
airports_extended['IATA'] = airports_extended['IATA'].fillna(airports_extended['IATA'].mode()[0])
airports_extended['ICAO'] = airports_extended['ICAO'].fillna(airports_extended['ICAO'].mode()[0])
print("Missing values after filling:")
print(pd.isna(airports_extended).sum()) # aantal missende waarden per kolom  


# In[389]:


schedule_airport = pd.read_csv('schedule_airport.csv')
schedule_airport.rename(columns={
    'STD': 'Datum',
    'FLT': 'Vlucht nummer',
    'STA_STD_ltc': 'Geplande aankomst',
    'ATA_ATD_ltc': 'Werkelijke aankomst',
    'LSV': 'L=inbound, S=outbound',
    'TAR': 'Geplande gate',
    'GAT': 'Werkelijke gate',
    'DL1': 'DL1',
    'IX1': 'IX1',
    'DL2': 'DL2',
    'IX2': 'IX2',
    'ACT': 'Vliegtuig type',
    'RWY': 'Landing/startbaan',
    'RWC': 'RWC',
    'Identifier': 'Identifier',
    'Org/Des': 'ICAO'
}, inplace=True)

schedule_airport.head(50)


# In[390]:


print(pd.isna(schedule_airport).sum()) # aantal missende waarden per kolom  


# In[395]:


# Fill missing values in 'Bestemming/afkomst' with the mode of the column
schedule_airport.fillna({'ICAO': schedule_airport['ICAO'].mode()[0]}, inplace=True)
print(pd.isna(schedule_airport).sum())


# In[396]:


vlucht1_sec = pd.read_excel('1Flight 1.xlsx')
vlucht2_sec = pd.read_excel('1Flight 2.xlsx')
vlucht3_sec = pd.read_excel('1Flight 3.xlsx')
vlucht4_sec = pd.read_excel('1Flight 4.xlsx')
vlucht5_sec = pd.read_excel('1Flight 5.xlsx')
vlucht6_sec = pd.read_excel('1Flight 6.xlsx')
vlucht7_sec = pd.read_excel('1Flight 7.xlsx')
pd.set_option("display.max_rows", None)
vlucht2_sec.head(80)


# In[397]:


print(pd.isna(vlucht2_sec).sum())


# In[400]:


vlucht2_sec = vlucht2_sec.dropna()
vlucht3_sec = vlucht3_sec.dropna()
vlucht4_sec = vlucht4_sec.dropna()
vlucht5_sec = vlucht5_sec.dropna()
vlucht6_sec = vlucht6_sec.dropna()
vlucht7_sec = vlucht7_sec.dropna()


# In[401]:


print(pd.isna(vlucht7_sec).sum())


# In[402]:


datasets = [vlucht1_sec, vlucht2_sec, vlucht3_sec, vlucht4_sec, vlucht5_sec, vlucht6_sec, vlucht7_sec]
for i in range(len(datasets)):
    datasets[i].columns = datasets[i].columns.str.replace(r'[\[\]]', '', regex=True)


# In[403]:


vlucht1_sec.head()


# ## Data COMBINEREN

# #### Airports Extended

# In[423]:


pd.set_option('display.max_columns', None) # heirmee krijg je allen kolomen van je dataset in oandas te zien 

merged_dataset = schedule_airport.merge(airports_extended,on='ICAO', how='inner')
# Verwijdert rijen met NaN-waarden
merged_dataset = merged_dataset.dropna()

merged_dataset.head(90)


# In[424]:


print(pd.isna(merged_dataset).sum())


# In[425]:


#pip install streamlit # Zorg ervoor dat st niet overschreven wordt


# In[426]:


import streamlit as st

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #0A1172;  /* Donkerblauwe kleur */
        font-family: 'Arial', sans-serif;  /* Je kunt het lettertype hier veranderen */
    }
    </style>
    <h1 class="title">Voorspelling & Analyse van vluchtvertraging ter wereld</h1>
    """, unsafe_allow_html=True)

st.write("""
    Welkom bij onze vluchtanalyse tool! Deze app biedt een uitgebreide kijk op de vluchtgegevens van verschillende luchthavens wereldwijd. Hier kunt u visualisaties verkennen die vertragingen, seizoensgebonden trends, en meer laten zien. Door verschillende datasets te combineren, krijgen we diepgaand inzicht in de vluchtprestaties en hun dynamiek. Ontdek hoe vliegtuigtypes, luchthavens en andere factoren bijdragen aan de algehele ervaring van luchtvaartpassagiers.
    """)

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)



# In[427]:


import plotly.express as px
merged_dataset['Werkelijke aankomst'] = pd.to_datetime(merged_dataset['Werkelijke aankomst'], errors='coerce')
merged_dataset['Geplande aankomst'] = pd.to_datetime(merged_dataset['Geplande aankomst'], errors='coerce')
merged_dataset['vertraging in (min)'] = (merged_dataset['Werkelijke aankomst'] - merged_dataset['Geplande aankomst']).dt.total_seconds() / 60

positive_delays = merged_dataset[merged_dataset['vertraging in (min)'] > 0]

top_airports = positive_delays.groupby('Airport')['vertraging in (min)'].sum().reset_index()

top_airports = top_airports.sort_values(by='vertraging in (min)', ascending=False).head(10)

fig = px.bar(top_airports, 
             x='Airport', 
             y='vertraging in (min)', 
             color='Airport',  # Elke luchthaven krijgt een unieke kleur
             title='Top 10 Luchthavens met Meeste Vertragingen (Positieve Vertragingen)', 
             labels={'Airport': 'Luchthaven', 'Vertraging in (min)': 'Totale Vertraging (minuten)'})

fig.update_layout(legend_title_text='Luchthaven')

st.plotly_chart(fig)
st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[428]:


import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

st.subheader("Aantal Vluchten per Luchthaven")

merged_dataset['Latitude'] = merged_dataset['Latitude'].astype(str).str.replace(',', '.')
merged_dataset['Longitude'] = merged_dataset['Longitude'].astype(str).str.replace(',', '.')

merged_dataset['Latitude'] = merged_dataset['Latitude'].astype(float)
merged_dataset['Longitude'] = merged_dataset['Longitude'].astype(float)

vluchten_per_luchthaven = merged_dataset.groupby('Airport')['Vlucht nummer'].count()

continenten = {
    'Wereldwijd': [0, 0, 2],
    'Europa': [54.5260, 15.2551, 4],
    'Afrika': [-8.7832, 34.5085, 3],
    'Noord-Amerika': [54.5260, -105.2551, 3],
    'Zuid-Amerika': [-14.2350, -51.9253, 3],
    'Azië': [34.0479, 100.6197, 3],
    'Oceanië': [-22.7359, 140.0188, 4]
}

continent = st.selectbox("Selecteer een continent", list(continenten.keys()))

def bepaal_kleur(aantal_vluchten):
    if aantal_vluchten < 2000:
        return 'darkgreen'  # Klein aantal vluchten
    elif 2000 <= aantal_vluchten < 4000:
        return 'orange'  # Gemiddeld aantal vluchten
    elif 4000 <= aantal_vluchten < 6000:
        return 'red' 
    else:
        return 'maroon'  

m = folium.Map(location=continenten[continent][:2], zoom_start=continenten[continent][2], 
               tiles='OpenStreetMap.DE', zoom_control=True, scrollWheelZoom=True, dragging=True)

for airport, aantal_vluchten in vluchten_per_luchthaven.items():
    locatie = merged_dataset[merged_dataset['Airport'] == airport][['Latitude', 'Longitude']].iloc[0]
    
    folium.CircleMarker(
        location=[locatie['Latitude'], locatie['Longitude']],
        radius=8, 
        color=bepaal_kleur(aantal_vluchten),
        fill=True,
        fill_opacity=0.7,
        fill_color=bepaal_kleur(aantal_vluchten) 
    ).add_to(m)

legend_html = '''
     <div style="position: fixed; 
                  bottom: 50px; left: 50px; width: 200px; height: auto; 
                  z-index:9999; font-size:10px; background-color:white;
                  border:2px solid grey; padding: 10px;">
     <h4>Legenda</h4>
     <i style="background: darkgreen; padding: 3px; border-radius: 5px;"></i> Klein aantal vluchten (0 - 2000)<br>
     <i style="background: orange; padding: 3px; border-radius: 5px;"></i> Gemiddeld aantal vluchten (2000 - 4000)<br>
     <i style="background: red; padding: 3px; border-radius: 5px;"></i> Groot aantal vluchten (4000 - 6000)<br>
     <i style="background: maroon; padding: 3px; border-radius: 5px;"></i> Zeer groot aantal vluchten (6000+)<br>
     </div>
     '''
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=700, returned_objects=[])

st.markdown(
    """
    <div style='border: 2px solid grey; padding: 10px;'>
        <h4>Legenda</h4>
        <span style="color: darkgreen; font-weight: bold;">&#9679;</span> Klein aantal vluchten (0 - 2000)<br>
        <span style="color: orange; font-weight: bold;">&#9679;</span> Gemiddeld aantal vluchten (2000 - 4000)<br>
        <span style="color: red; font-weight: bold;">&#9679;</span> Groot aantal vluchten (4000 - 6000)<br>
        <span style="color: maroon; font-weight: bold;">&#9679;</span> Zeer groot aantal vluchten (6000+)<br>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[430]:


import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
st.subheader("Weergave vertragingen per luchthaven per land")

merged_dataset['Geplande aankomst'] = pd.to_datetime(merged_dataset['Geplande aankomst'], errors='coerce')
merged_dataset['Werkelijke aankomst'] = pd.to_datetime(merged_dataset['Werkelijke aankomst'], errors='coerce')

if merged_dataset['Geplande aankomst'].isnull().any() or merged_dataset['Werkelijke aankomst'].isnull().any():
    st.warning("Er zijn datums die niet konden worden geconverteerd. Controleer je dataset.")

merged_dataset['vertraging in (min)'] = (merged_dataset['Werkelijke aankomst'] - merged_dataset['Geplande aankomst']).dt.total_seconds() / 60

merged_dataset = merged_dataset[(merged_dataset['vertraging in (min)'] > 0) & (merged_dataset['vertraging in (min)'] <= 300)]

filter_by_country = st.checkbox('Filteren op landen', key='filter_country')

if filter_by_country:
    selected_country = st.selectbox('Selecteer een land:', merged_dataset['Country'].unique(), key='country_selector')
    filtered_data = merged_dataset[merged_dataset['Country'] == selected_country]
else:
    filtered_data = merged_dataset

unique_airports = filtered_data['Airport'].nunique()
selected_palette = sns.color_palette("husl", unique_airports)

if not filtered_data.empty:
    plt.figure(figsize=(12, 8))
    ax = sns.stripplot(x='vertraging in (min)', y='Airport', data=filtered_data, hue='Airport',
                       palette=selected_palette, linewidth=0.5, jitter=True, legend=False)
    plt.xlabel('Vertraging in minuten')
    plt.ylabel('Luchthaven')
    plt.title(f'Vertraging per luchthaven - {selected_country if filter_by_country else "Alle luchthavens"}')
    ax.set_xlim(left=0)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(['{:2.0f}m'.format(x) for x in ax.get_xticks()])

    st.pyplot(plt)

    mean_delay = filtered_data['vertraging in (min)'].mean()
    median_delay = filtered_data['vertraging in (min)'].median()
    max_delay = filtered_data['vertraging in (min)'].max()

    st.write(f"Gemiddelde vertraging van luchthavens in {selected_country if filter_by_country else 'Alle luchthavens'}: **{mean_delay:.2f} minuten**")
    st.write(f"Maximale vertraging: **{max_delay:.2f} minuten**")


    if mean_delay > 30:
        st.write("Dit geeft aan dat de gemiddelde vertraging boven de 30 minuten ligt, wat een probleem kan zijn voor de passagierservaring.")
    else:
        st.write("De gemiddelde vertraging is relatief laag, wat wijst op een goede punctualiteit van de luchthavens.")

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)



# In[434]:


import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import uuid  
st.subheader("Seizoensgebonden Trends in Aantal Vluchten en Vertragingen")

merged_dataset['Datum'] = pd.to_datetime(merged_dataset['Datum'], dayfirst=True, errors='coerce')

merged_dataset['Maand'] = merged_dataset['Datum'].dt.to_period('M')

filter_op_land = st.checkbox('Filteren op Land', key='filter_land_checkbox_unique')

if filter_op_land:
    landen = merged_dataset['Country'].unique()  
    gekozen_land = st.selectbox('Selecteer een land:', landen, key='land_selectbox')
    merged_dataset = merged_dataset[merged_dataset['Country'] == gekozen_land]

flight_counts = merged_dataset.groupby('Maand').size().reset_index(name='Aantal Vluchten')
delay_data = merged_dataset[(merged_dataset['Werkelijke aankomst'].notna()) & (merged_dataset['Geplande aankomst'].notna())]
delay_data['Vertraging (min)'] = (delay_data['Werkelijke aankomst'] - delay_data['Geplande aankomst']).dt.total_seconds() / 60
average_delays = delay_data.groupby('Maand')['Vertraging (min)'].mean().reset_index()

flight_counts['Maand'] = flight_counts['Maand'].dt.strftime('%m/%y')  # Maand als '01/19'
average_delays['Maand'] = average_delays['Maand'].dt.strftime('%m/%y')  # Maand als '01/19'

fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.set_xlabel('Maand')
ax1.set_ylabel('Aantal Vluchten', color='tab:blue')
ax1.plot(flight_counts['Maand'], flight_counts['Aantal Vluchten'], color='tab:blue', marker='o', label='Aantal Vluchten')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Gemiddelde Vertraging (minuten)', color='tab:red')
ax2.plot(average_delays['Maand'], average_delays['Vertraging (min)'], color='tab:red', marker='o', label='Gemiddelde Vertraging')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Seizoensgebonden Trends in Aantal Vluchten en Gemiddelde Vertragingen')
fig.tight_layout()  
st.pyplot(fig)

if not flight_counts.empty and not average_delays.empty:
    
    maand_max_vluchten = flight_counts.loc[flight_counts['Aantal Vluchten'].idxmax()]
    maand_max_vertraging = average_delays.loc[average_delays['Vertraging (min)'].idxmax()]
    
    st.write(f"In de grafiek is te zien dat het aantal vluchten in de maand **{maand_max_vluchten['Maand']}** het hoogst is, met **{maand_max_vluchten['Aantal Vluchten']}** vluchten.")
    st.write(f"Bovendien blijkt dat de maand **{maand_max_vertraging['Maand']}** de hoogste gemiddelde vertraging had, met een gemiddelde vertraging van **{maand_max_vertraging['Vertraging (min)']:.2f}** minuten.")
    
    st.write("Deze trends kunnen waardevolle inzichten bieden voor zowel luchtvaartmaatschappijen als reizigers, omdat ze wijzen op periodes met hogere drukte en mogelijke vertragingen. "
             "Het is belangrijk om deze informatie in overweging te nemen bij het plannen van reizen, vooral tijdens drukke seizoenen.")

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[435]:


st.subheader("Relatie tussen Luchthaven Drukte en Vertragingen")

merged_dataset['Aantal Wijzigingen Gate'] = (merged_dataset['Geplande gate'] != merged_dataset['Werkelijke gate']).astype(int)

luchthaven_stats = merged_dataset.groupby('Airport').agg({
    'vertraging in (min)': 'mean',  
    'Vlucht nummer': 'count',       
    'Aantal Wijzigingen Gate': 'sum'  
}).reset_index()

luchthaven_stats = luchthaven_stats[luchthaven_stats['Airport'] != '']

luchthaven_stats = luchthaven_stats.sort_values(by='Aantal Wijzigingen Gate', ascending=False)
max_luchthavens = st.slider('Aantal luchthavens om weer te geven', min_value=1, max_value=len(luchthaven_stats), value=10)
luchthaven_stats = luchthaven_stats.head(max_luchthavens)

fig = px.scatter(luchthaven_stats, x='Vlucht nummer', y='vertraging in (min)', 
                 size='Aantal Wijzigingen Gate', color='Airport', 
                 labels={'Vlucht nummer': 'Aantal Vluchten', 'vertraging in (min)': 'Gemiddelde Vertraging (min)'},
                 hover_name='Airport')

st.plotly_chart(fig)

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[436]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.subheader('Aantal vluchten per luchthaven tijdens vakantieperiode gedurende jaren 2019 en 2020')
def plot_vluchten_per_maand(filtered_df):
    filtered_df['Datum'] = pd.to_datetime(filtered_df['Datum'])
    filtered_df['Month'] = filtered_df['Datum'].dt.month
    filtered_df['Year'] = filtered_df['Datum'].dt.year

    monthly_flights = filtered_df.groupby(['Year', 'Month']).size().reset_index(name='Aantal Vluchten')
    
    all_months = monthly_flights.pivot(index='Month', columns='Year', values='Aantal Vluchten').fillna(0)
    fig, ax = plt.subplots()
    all_months.plot(ax=ax, marker='o')

    ax.set_title('Aantal Vluchten per Maand')
    ax.set_xlabel('Maand')
    ax.set_ylabel('Aantal Vluchten')
    ax.set_xticks(range(1, 13))  # Zet de x-ticks op maanden
    ax.set_xticklabels(['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'])

    for month in range(1, 13):
        if month in [7, 8, 12]: 
            ax.axvspan(month - 0.5, month + 0.5, color='yellow', alpha=0.3)

    st.pyplot(fig)

filter_landen = st.checkbox('Filter op landen', key='filter_landen_checkbox')

if filter_landen:
    landen = merged_dataset['Country'].unique()
    geselecteerd_land = st.selectbox('Selecteer een land', landen, key='land_selectbox')

    filtered_df = merged_dataset[merged_dataset['Country'] == geselecteerd_land]
else:
    filtered_df = merged_dataset

plot_vluchten_per_maand(filtered_df)
st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[437]:


import streamlit as st
import pandas as pd
import plotly.express as px

st.subheader("Gemiddelde Vertraging per Tijdstip van de Dag per land en luchthaven ")

def get_time_of_day(hour):
    if hour < 6:
        return 'Nacht'
    elif hour < 12:
        return 'Ochtend'
    elif hour < 18:
        return 'Middag'
    else:
        return 'Avond'

merged_dataset['Tijd van de Dag'] = merged_dataset['Geplande aankomst'].dt.hour.apply(get_time_of_day)
positieve_vertraging = merged_dataset[merged_dataset['vertraging in (min)'] > 0]
gemiddelde_vertraging = positieve_vertraging.groupby(['Country', 'Airport', 'Tijd van de Dag'])['vertraging in (min)'].mean().reset_index()

landen = merged_dataset['Country'].unique()
gekozen_land = st.selectbox('Kies een land:', landen)

luchtvaartmaatschappijen = merged_dataset[merged_dataset['Country'] == gekozen_land]['Airport'].unique()
gekozen_luchthaven = st.selectbox('Kies een luchthaven:', luchtvaartmaatschappijen)

filtered_data = gemiddelde_vertraging[(gemiddelde_vertraging['Country'] == gekozen_land) & 
                                       (gemiddelde_vertraging['Airport'] == gekozen_luchthaven)]

if not filtered_data.empty:
    fig = px.line(filtered_data, x='Tijd van de Dag', y='vertraging in (min)',
                  labels={'Vertraging in (min)': 'Gemiddelde Vertraging (minuten)', 'Tijd van de Dag': 'Tijd van de Dag'},
                  markers=True)

    st.plotly_chart(fig)
else:
    st.write("Geen gegevens beschikbaar voor de geselecteerde luchthaven.")

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[438]:


st.subheader("Aantal Gate Wijzigingen per Luchthaven")

filter_op_land = st.checkbox('Filteren op Land')

if filter_op_land:
    landen = merged_dataset['Country'].unique()  # Unieke landen ophalen
    gekozen_land = st.selectbox('Selecteer een land:', landen)
    merged_dataset = merged_dataset[merged_dataset['Country'] == gekozen_land]

merged_dataset['Aantal Wijzigingen Gate'] = merged_dataset['Geplande gate'] != merged_dataset['Werkelijke gate']
merged_dataset['Aantal Wijzigingen Gate'] = merged_dataset['Aantal Wijzigingen Gate'].astype(int)
aantal_wijzigingen = merged_dataset.groupby('Airport')['Aantal Wijzigingen Gate'].sum().reset_index()

aantal_wijzigingen = aantal_wijzigingen[aantal_wijzigingen['Airport'] != '']

aantal_wijzigingen = aantal_wijzigingen.sort_values(by='Aantal Wijzigingen Gate', ascending=False)

bin_count = st.slider('Selecteer aantal bins:', min_value=5, max_value=50, value=10)
fig, ax = plt.subplots(figsize=(14, 7))
ax.barh(aantal_wijzigingen['Airport'].head(bin_count), aantal_wijzigingen['Aantal Wijzigingen Gate'].head(bin_count), color='skyblue')
ax.set_xlabel('Aantal Wijzigingen in Gate')
ax.set_title('Histogram van Aantal Wijzigingen in Gate per Luchthaven (Groot naar Klein)')

st.pyplot(fig)

st.write("Dit histogram toont het aantal wijzigingen in gates per luchthaven, gesorteerd van klein naar groot. "
         "De luchthavens met de meeste wijzigingen zijn vaak de drukste en bekendste luchthavens in het land, "
         "wat kan wijzen op operationele uitdagingen of een hogere verkeersdrukte. "
         "Door deze gegevens te analyseren, kunnen luchtvaartmaatschappijen en luchthavenbeheer inzicht krijgen in hun operationele efficiëntie. ")

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[439]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.subheader("Relatie tussen Gate Wijzigingen en Vertragingen per Luchthaven")

countries = merged_dataset['Country'].unique()
selected_country = st.selectbox("Kies een land:", countries, key="country_selectbox")

filtered_data = merged_dataset[merged_dataset['Country'] == selected_country]
grouped_data = filtered_data.groupby('Airport').agg({
    'vertraging in (min)': 'mean', 
    'Aantal Wijzigingen Gate': 'sum'  
}).reset_index()


fig, ax = plt.subplots(figsize=(12, 6))

if not grouped_data.empty:
    scatter = ax.scatter(
        grouped_data['vertraging in (min)'],  
        grouped_data['Aantal Wijzigingen Gate'],  
        c=grouped_data.index,  
        cmap='tab10',  
        alpha=0.7
    )
    
    ax.set_xlabel('Gemiddelde Vertragingen (minuten)')
    ax.set_ylabel('Totale Aantal Gate Wijzigingen')
    ax.set_title(f'Relatie tussen Gate Wijzigingen en Vertragingen per Luchthaven voor {selected_country}')

    unique_airports = grouped_data['Airport'].unique()
    handles, _ = scatter.legend_elements(prop="colors") 
    ax.legend(handles, unique_airports, title="Luchthaven", bbox_to_anchor=(1.05, 1), loc='upper left')

    st.pyplot(fig)

st.write("De scatterplot toont de relatie tussen gemiddelde vertragingen (in minuten) en het aantal gate wijzigingen per luchthaven in {selected_country}. "
         "Elk punt vertegenwoordigt een luchthaven. Een hogere gemiddelde vertraging lijkt samen te gaan met een groter aantal gate wijzigingen. ")

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[440]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.subheader("Vertraging in minuten per vliegtuigtype")
top_n = st.slider('Aantal vliegtuigtypes om te tonen', min_value=5, max_value=50, step=5, value=10)

filtered_data = merged_dataset[merged_dataset['vertraging in (min)'] > 0]

top_vliegtuigen = filtered_data['Vliegtuig type'].value_counts()

top_vliegtuigen_list = top_vliegtuigen.head(top_n).index
top_data = filtered_data[filtered_data['Vliegtuig type'].isin(top_vliegtuigen_list)]
plt.figure(figsize=(12, 6))
sns.violinplot(x='Vliegtuig type', y='vertraging in (min)', data=top_data)
plt.xticks(rotation=45)
plt.title(f'Vertraging per Vliegtuig Type (Top {top_n} Vliegtuigen)')
st.pyplot(plt)

st.write('De violenplot toont de verdeling van vertragingen per vliegtuigtype. De breedte van elke "viool" geeft aan waar de meeste vertragingen voorkomen (dichtheid).')
st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[441]:


import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.subheader('Vlucht Afstanden en Routes Overzicht')

def v_direct(start, end):
    lat1, lon1 = np.radians(start)
    lat2, lon2 = np.radians(end)
    dlon = np.radians(end[1] - start[1])
    a = 6371.0  # Aardstraal in kilometers
    c = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    distance = a * (2 * np.arctan2(np.sqrt(c), np.sqrt(1 - c)))
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    azimuth = np.degrees(np.arctan2(x, y))
    azimuth = (azimuth + 360) % 360
    return distance, azimuth

vlucht1_sec = pd.read_excel('1Flight 1.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht2_sec = pd.read_excel('1Flight 2.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht3_sec = pd.read_excel('1Flight 3.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht4_sec = pd.read_excel('1Flight 4.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht5_sec = pd.read_excel('1Flight 5.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht6_sec = pd.read_excel('1Flight 6.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht7_sec = pd.read_excel('1Flight 7.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])

vluchten = [vlucht1_sec, vlucht2_sec, vlucht3_sec, vlucht4_sec, vlucht5_sec, vlucht6_sec, vlucht7_sec]
kleuren = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

tabs = st.tabs([f"Vlucht {i+1}" for i in range(len(vluchten))])

for i, vlucht in enumerate(vluchten):
    with tabs[i]:
        start_point = (vlucht['[3d Latitude]'].iloc[0], vlucht['[3d Longitude]'].iloc[0])
        end_point = (vlucht['[3d Latitude]'].iloc[-2], vlucht['[3d Longitude]'].iloc[-2])

        distance, azimuth = v_direct(start_point, end_point)

        m = folium.Map(location=[52.0, 5.0], zoom_start=5)
        folium.Marker(location=start_point, popup=f'Vlucht {i+1} Start', icon=folium.Icon(color=kleuren[i])).add_to(m)
        folium.Marker(location=end_point, popup=f'Vlucht {i+1} Eind', icon=folium.Icon(color=kleuren[i])).add_to(m)
        folium.PolyLine(locations=[start_point, end_point], color=kleuren[i], weight=2.5, opacity=0.7).add_to(m)

        mid_point = [(start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2]
        folium.Marker(location=mid_point, popup=f'Afstand: {distance:.2f} km, Azimut: {azimuth:.2f} graden').add_to(m)

        # Toon de kaart in de betreffende tab
        st_folium(m, width=700, height=500)

vlucht_data = []
for i, vlucht in enumerate(vluchten):
    start_point = (vlucht['[3d Latitude]'].iloc[0], vlucht['[3d Longitude]'].iloc[0])
    end_point = (vlucht['[3d Latitude]'].iloc[-2], vlucht['[3d Longitude]'].iloc[-2])
    distance, _ = v_direct(start_point, end_point)
    start_time = pd.to_datetime(vlucht['Time (secs)'].iloc[0], unit='s')
    end_time = pd.to_datetime(vlucht['Time (secs)'].iloc[-2], unit='s')
    flight_duration = end_time - start_time
    hours, remainder = divmod(flight_duration.total_seconds(), 3600)
    minutes, _ = divmod(remainder, 60)
    vlucht_data.append([f'Vlucht {i+1}', start_point, end_point, f'{distance:.2f} km', f'{int(hours)} uur {int(minutes)} min'])

vlucht_df = pd.DataFrame(vlucht_data, columns=['Vlucht', 'Startlocatie', 'Eindlocatie', 'Afstand', 'Duur'])
st.dataframe(vlucht_df.head(7))

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)



# In[442]:


import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.subheader('Vergelijking van Vluchtduur op basis van Snelheid')

vlucht1_sec = pd.read_excel('1Flight 1.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht2_sec = pd.read_excel('1Flight 2.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht3_sec = pd.read_excel('1Flight 3.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht4_sec = pd.read_excel('1Flight 4.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht5_sec = pd.read_excel('1Flight 5.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht6_sec = pd.read_excel('1Flight 6.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])
vlucht7_sec = pd.read_excel('1Flight 7.xlsx').dropna(subset=['[3d Latitude]', '[3d Longitude]'])

vluchten = {
    'Vlucht 1': vlucht1_sec,
    'Vlucht 2': vlucht2_sec,
    'Vlucht 3': vlucht3_sec,
    'Vlucht 4': vlucht4_sec,
    'Vlucht 5': vlucht5_sec,
    'Vlucht 6': vlucht6_sec,
    'Vlucht 7': vlucht7_sec
}

selected_flights = st.multiselect(
    'Selecteer de vluchten die je wilt vergelijken',
    options=list(vluchten.keys()),
    default=['Vlucht 1', 'Vlucht 2'],
    key='flight_selection' 
)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours:02}h {minutes:02}min"

if selected_flights:
    fig = go.Figure()

    max_snelheden = []
    vlucht_duren = []

    kleuren = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue']
    
    for idx, vlucht in enumerate(selected_flights):
        vlucht_data = vluchten[vlucht]
        tijd_in_seconden = vlucht_data['Time (secs)'] - vlucht_data['Time (secs)'].min()
        
        vlucht_data['TRUE AIRSPEED (derived)'] = pd.to_numeric(vlucht_data['TRUE AIRSPEED (derived)'], errors='coerce')
        snelheid = vlucht_data['TRUE AIRSPEED (derived)']
        
        fig.add_trace(go.Scatter(
            x=tijd_in_seconden,
            y=snelheid,
            mode='lines+markers',
            name=f'{vlucht} - Snelheid (TAS)',
            line=dict(color=kleuren[idx % len(kleuren)]),
            marker=dict(size=4),
        ))

        eindpunt_x = tijd_in_seconden.iloc[-1]
        eindpunt_y = snelheid.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[eindpunt_x],
            y=[eindpunt_y],
            mode='markers+text',
            marker=dict(color=kleuren[idx % len(kleuren)], size=10),
            text=[format_time(eindpunt_x)],
            textposition='top right',
            showlegend=False
        ))

        max_snelheden.append((vlucht, snelheid.max()))
        vlucht_duren.append((vlucht, eindpunt_x))

    fig.update_layout(
        xaxis_title="Tijd (hh:mm)",
        xaxis_tickvals=[0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200, 7800, 8400, 9000],
        xaxis_ticktext=[format_time(t) for t in [0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200, 7800, 8400, 9000]],
        yaxis_title="Snelheid (True Airspeed, TAS)",
        legend=dict(x=0.1, y=1.1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


    for vlucht, max_snelheid in max_snelheden:
        duur = next(duur for v, duur in vlucht_duren if v == vlucht)
        st.write(f"{vlucht} heeft een maximale snelheid van {max_snelheid:.2f} knopen bereikt, "
                 f"en duurde in totaal {format_time(duur)}.")

    snelste_vlucht = max(max_snelheden, key=lambda x: x[1])
    langste_vlucht = max(vlucht_duren, key=lambda x: x[1])

    st.write(f"De snelste vlucht was **{snelste_vlucht[0]}** met een maximale snelheid van **{snelste_vlucht[1]:.2f} knopen**.")
    st.write(f"De langste vlucht was **{langste_vlucht[0]}** met een duur van **{format_time(langste_vlucht[1])}**.")

else:
    st.write('Selecteer ten minste één vlucht om de grafiek te zien.')

st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[443]:


import streamlit as st
import pandas as pd

merged_dataset.columns = merged_dataset.columns.str.strip()
st.subheader('Vluchtinformatie')

flight_number = st.text_input('Voer hier je vluchtnummer in')
if flight_number:
    flight_info = merged_dataset[merged_dataset['Vlucht nummer'] == flight_number]
    if not flight_info.empty:
        st.subheader(f"Vluchtinformatie voor {flight_number}")

        flight_date = pd.to_datetime(flight_info.iloc[0]['Datum']).strftime('%d/%m/%Y')

        arrival_time = pd.to_datetime(flight_info.iloc[0]['Werkelijke aankomst']).strftime('%H:%M')

        st.write(f"**Datum:** {flight_date}")
        st.write(f"**Werkelijke Gate:** {flight_info.iloc[0]['Werkelijke gate']}")
        st.write(f"**Werkelijke Aankomsttijd:** {arrival_time}")
    else:
        st.error(f"Geen informatie gevonden voor vlucht {flight_number}")
else:
    st.info("Voer hier je vluchtnummer in")
st.markdown("<hr style='border: 3px solid darkblue;'>", unsafe_allow_html=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




