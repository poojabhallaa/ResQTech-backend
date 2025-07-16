# import streamlit as st
# import pickle
# import requests
# import pandas as pd
# import folium
# from streamlit_folium import st_folium

# # Load model
# with open('disaster_relief_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# st.title("🌍 Disaster Relief Prediction System")

# # ----------- Disaster Info Input ------------
# st.subheader("📌 Step 1: Enter Disaster Info")

# with st.form("disaster_form"):
#     disaster_type = st.selectbox("Disaster Type", ["Earthquake", "Flood", "Cyclone", "Landslide", "Fire"])
#     severity = st.selectbox("Severity Level", ["Low", "Moderate", "High"])
#     location = st.text_input("Location (City or Country)")  # User inputs location here
#     submitted_disaster = st.form_submit_button("Submit Disaster Info")

# if submitted_disaster:
#     reliefweb_url = "https://api.reliefweb.int/v1/disasters"
#     reliefweb_params = {
#         'limit': 1,
#         'query[value]': disaster_type,
#         'query[operator]': 'AND'
#     }
#     response = requests.get(reliefweb_url, params=reliefweb_params)
#     data = response.json()

#     if data.get('data'):
#         fields = data['data'][0]['fields']
#         disaster_name = fields.get('name', 'N/A')
#         country_name = fields.get('country', [{}])[0].get('name', 'Unknown')

#         location_display = location or country_name

#         st.session_state.disaster_data = {
#             'name': disaster_name,
#             'disaster_type': disaster_type,
#             'severity': severity,
#             'location': location_display
#         }
#         st.success("✅ Disaster info fetched!")
#     else:
#         st.warning("No disaster info found from ReliefWeb.")

# # ----------- Weather Info Input ------------
# st.subheader("🌤️ Step 2: Get Weather Info")

# with st.form("weather_form"):
#     city_name = st.text_input("City Name (for Weather Info)")  # User inputs city name
#     submitted_weather = st.form_submit_button("Submit Weather Info")

# if submitted_weather:
#     try:
#         weather_api_key = st.secrets["weather_api_key"]
#     except:
#         weather_api_key = "7eb20794085a2d30dadd96937ed3e87c"  # Replace with your real key

#     weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={weather_api_key}"
#     response = requests.get(weather_url)
#     weather_data = response.json()

#     if 'main' in weather_data and 'weather' in weather_data:
#         temperature = weather_data['main']['temp'] - 273.15
#         weather_condition = weather_data['weather'][0]['description']
#         lat = weather_data['coord']['lat']
#         lon = weather_data['coord']['lon']

#         st.session_state.weather = {
#             'temperature': temperature,
#             'weather_condition': weather_condition,
#             'lat': lat,
#             'lon': lon,
#             'city': city_name
#         }
#         st.success("✅ Weather info fetched!")
#     else:
#         st.warning("Failed to fetch weather info!")

# # ----------- Prediction ---------------------
# st.subheader("📈 Step 3: Predict Resources")

# if st.button("Predict Relief Requirements"):
#     if "disaster_data" in st.session_state and "weather" in st.session_state:
#         input_df = pd.DataFrame({
#             'disaster_type': [st.session_state.disaster_data['disaster_type']],
#             'location': [st.session_state.disaster_data['location']],
#             'severity_level': [st.session_state.disaster_data['severity']],
#             'weather_condition': [st.session_state.weather['weather_condition']],
#             'temperature': [st.session_state.weather['temperature']],
#             'road_blocked': [0],
#             'population_affected': [0],
#             'medical_cases': [0],
#             'infrastructure_damage_percent': [0]
#         })

#         prediction = model.predict(input_df)

#         st.session_state.prediction_result = {
#             'food_required': prediction[0][0],
#             'rescue_teams': prediction[0][1],
#             'medical_aid': prediction[0][2]
#         }

# # Display prediction and map if available in session_state
# if "prediction_result" in st.session_state:
#     result = st.session_state.prediction_result

#     st.success("🎯 Prediction Complete!")
#     st.write(f"**Disaster Name:** {st.session_state.disaster_data['name']}")
#     st.write(f"**Location:** {st.session_state.disaster_data['location']}")
#     st.write(f"**Temperature:** {st.session_state.weather['temperature']:.2f} °C")
#     st.write(f"**Weather Condition:** {st.session_state.weather['weather_condition']}")

#     st.markdown("---")
#     st.subheader("🔍 Prediction Results")
#     st.write(f"🛒 **Food Required (Tons):** {result['food_required']:.2f}")
#     st.write(f"🚑 **Rescue Teams Required:** {result['rescue_teams']:.2f}")
#     st.write(f"🩺 **Medical Aid Required:** {result['medical_aid']:.2f}")

#     # -------- Weather Map -----------
#     st.subheader("🗺️ Weather Map of Location")

#     m = folium.Map(location=[st.session_state.weather['lat'], st.session_state.weather['lon']], zoom_start=8)
#     popup_text = f"{st.session_state.weather['city']}<br>Temp: {st.session_state.weather['temperature']:.2f}°C<br>{st.session_state.weather['weather_condition'].title()}"
#     folium.Marker(
#         [st.session_state.weather['lat'], st.session_state.weather['lon']],
#         popup=popup_text,
#         tooltip="Weather Info",
#         icon=folium.Icon(color="blue", icon="cloud")
#     ).add_to(m)

#     st_folium(m, width=700, height=450)
# else:
#     st.info("👉 Please click the prediction button to see results and map.")

# import streamlit as st
# import pandas as pd
# import pickle
# import requests
# import folium
# from streamlit_folium import st_folium

# # Load model
# with open('disaster_relief_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# # -------- Sidebar Option ---------
# st.sidebar.title("🔧 Prediction Method")
# method = st.sidebar.selectbox("Choose Prediction Type", ["Traditional", "API-Based"])

# st.title("🌍 Disaster Relief Prediction System")

# # ------------ Traditional Method (Default) ------------
# if method == "Traditional":
#     st.subheader("📌 Step 1: Enter Disaster Info (Traditional Method)")

#     with st.form("disaster_form"):
#         disaster_type = st.selectbox("Disaster Type", ["Earthquake", "Flood", "Cyclone", "Landslide", "Fire"])
#         severity = st.selectbox("Severity Level", ["Low", "Moderate", "High"])
#         location = st.text_input("Location (City or Country)")
#         submitted_disaster = st.form_submit_button("Submit Disaster Info")

#     if submitted_disaster:
#         reliefweb_url = "https://api.reliefweb.int/v1/disasters"
#         reliefweb_params = {
#             'limit': 1,
#             'query[value]': disaster_type,
#             'query[operator]': 'AND'
#         }
#         response = requests.get(reliefweb_url, params=reliefweb_params)
#         data = response.json()

#         if data.get('data'):
#             fields = data['data'][0]['fields']
#             disaster_name = fields.get('name', 'N/A')
#             country_name = fields.get('country', [{}])[0].get('name', 'Unknown')

#             location_display = location or country_name

#             st.session_state.disaster_data = {
#                 'name': disaster_name,
#                 'disaster_type': disaster_type,
#                 'severity': severity,
#                 'location': location_display
#             }
#             st.success("✅ Disaster info fetched!")
#         else:
#             st.warning("No disaster info found from ReliefWeb.")

#     st.subheader("🌤️ Step 2: Get Weather Info")

#     with st.form("weather_form"):
#         city_name = st.text_input("City Name (for Weather Info)")
#         submitted_weather = st.form_submit_button("Submit Weather Info")

#     if submitted_weather:
#         try:
#             weather_api_key = st.secrets["weather_api_key"]
#         except:
#             weather_api_key = "7eb20794085a2d30dadd96937ed3e87c"

#         weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={weather_api_key}"
#         response = requests.get(weather_url)
#         weather_data = response.json()

#         if 'main' in weather_data and 'weather' in weather_data:
#             temperature = weather_data['main']['temp'] - 273.15
#             weather_condition = weather_data['weather'][0]['description']
#             lat = weather_data['coord']['lat']
#             lon = weather_data['coord']['lon']

#             st.session_state.weather = {
#                 'temperature': temperature,
#                 'weather_condition': weather_condition,
#                 'lat': lat,
#                 'lon': lon,
#                 'city': city_name
#             }
#             st.success("✅ Weather info fetched!")
#         else:
#             st.warning("Failed to fetch weather info!")

#     st.subheader("📈 Step 3: Predict Resources")

#     if st.button("Predict Relief Requirements"):
#         if "disaster_data" in st.session_state and "weather" in st.session_state:
#             input_df = pd.DataFrame({
#                 'disaster_type': [st.session_state.disaster_data['disaster_type']],
#                 'location': [st.session_state.disaster_data['location']],
#                 'severity_level': [st.session_state.disaster_data['severity']],
#                 'weather_condition': [st.session_state.weather['weather_condition']],
#                 'temperature': [st.session_state.weather['temperature']],
#                 'road_blocked': [0],
#                 'population_affected': [0],
#                 'medical_cases': [0],
#                 'infrastructure_damage_percent': [0]
#             })

#             prediction = model.predict(input_df)
#             food_required, rescue_teams, medical_aid = prediction[0]

#             # Save results in session state
#             st.session_state.prediction_results = {
#                 'food_required': food_required,
#                 'rescue_teams': rescue_teams,
#                 'medical_aid': medical_aid
#             }

#             st.success("🎯 Prediction Complete!")
#             st.write(f"**Disaster Name:** {st.session_state.disaster_data['name']}")
#             st.write(f"**Location:** {st.session_state.disaster_data['location']}")
#             st.write(f"**Temperature:** {st.session_state.weather['temperature']:.2f} °C")
#             st.write(f"**Weather Condition:** {st.session_state.weather['weather_condition']}")

#             st.markdown("---")
#             st.subheader("🔍 Prediction Results")
#             st.write(f"🛒 **Food Required (Tons):** {st.session_state.prediction_results['food_required']:.2f}")
#             st.write(f"🚑 **Rescue Teams Required:** {st.session_state.prediction_results['rescue_teams']:.2f}")
#             st.write(f"🩺 **Medical Aid Required:** {st.session_state.prediction_results['medical_aid']:.2f}")

#             st.subheader("🗺️ Weather Map of Location")
#             m = folium.Map(location=[st.session_state.weather['lat'], st.session_state.weather['lon']], zoom_start=8)
#             popup_text = f"{st.session_state.weather['city']}<br>Temp: {st.session_state.weather['temperature']:.2f}°C<br>{st.session_state.weather['weather_condition'].title()}"
#             folium.Marker(
#                 [st.session_state.weather['lat'], st.session_state.weather['lon']],
#                 popup=popup_text,
#                 tooltip="Weather Info",
#                 icon=folium.Icon(color="blue", icon="cloud")
#             ).add_to(m)
#             st_folium(m, width=700, height=450)
#         else:
#             st.error("❌ Please submit both Disaster Info and Weather Info first.")

# # ------------ API-Based / Real-Time Manual Input Method ------------
# elif method == "API-Based":
#     st.subheader("🧠 Real-Time Input (API-Based Manual Data Entry)")

#     # Initialize session state for button click
#     if "api_predict_clicked" not in st.session_state:
#         st.session_state.api_predict_clicked = False

#     disaster_type = st.selectbox("Disaster Type", ['earthquake', 'flood', 'hurricane'])
#     location = st.selectbox("Location", ['urban', 'rural'])
#     severity_level = st.selectbox("Severity Level", ['low', 'medium', 'high'])
#     weather_condition = st.selectbox("Weather Condition", ['sunny', 'rainy', 'stormy'])

#     road_blocked_input = st.selectbox("Road Blocked", ["Yes", "No"])
#     road_blocked = 1 if road_blocked_input == "Yes" else 0

#     population_affected = st.number_input("Population Affected", min_value=0)
#     medical_cases = st.number_input("Medical Cases", min_value=0)
#     infrastructure_damage_percent = st.slider("Infrastructure Damage (%)", 0, 100)

#     # Predict Button
#     if st.button("Predict Relief Requirements (Manual)"):
#         st.session_state.api_predict_clicked = True

#     # Reset Button
#     if st.button("🔄 Reset Prediction"):
#         st.session_state.api_predict_clicked = False

#     # Show Results if Button Was Clicked
#     if st.session_state.api_predict_clicked:
#         input_data = pd.DataFrame([{
#             'disaster_type': disaster_type,
#             'location': location,
#             'severity_level': severity_level,
#             'weather_condition': weather_condition,
#             'road_blocked': road_blocked,
#             'population_affected': population_affected,
#             'medical_cases': medical_cases,
#             'infrastructure_damage_percent': infrastructure_damage_percent
#         }])

#         prediction = model.predict(input_data)
#         food, rescue, medical = prediction[0]

#         # Save results in session state
#         st.session_state.api_prediction_results = {
#             'food': food,
#             'rescue': rescue,
#             'medical': medical
#         }

#         st.success("📦 Relief Prediction Results:")
#         st.write(f"🍚 Food Required (tons): **{st.session_state.api_prediction_results['food']:.2f}**")
#         st.write(f"🚑 Medical Aid Required: **{st.session_state.api_prediction_results['medical']:.2f}**")
#         st.write(f"🧑‍🚒 Rescue Teams Required: **{st.session_state.api_prediction_results['rescue']:.2f}**")





import streamlit as st
import pickle
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from google.generativeai import configure, GenerativeModel

# Configure Gemini AI
configure(api_key="AIzaSyA0Lw_Vpr2G1rF3ZkHfFZCgj9fhl8P1iYc")
genai_model = GenerativeModel("gemini-2.0-flash")

# Load model
with open('C:\\Users\\pooja\\Downloads\\ResQ-Tech\\backend\\disaster_relief_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🌍 Disaster Relief Advance System")

# ----------- Disaster Info Input ------------
st.subheader("📌 Step 1: Enter Disaster Info")

with st.form("disaster_form"):
    disaster_type = st.selectbox("Disaster Type", ["Earthquake", "Flood", "Cyclone", "Landslide", "Fire"])
    severity = st.selectbox("Severity Level", ["Low", "Moderate", "High"])
    location = st.text_input("Location (City or Country)")
    submitted_disaster = st.form_submit_button("Submit Disaster Info")

if submitted_disaster:
    reliefweb_url = "https://api.reliefweb.int/v1/disasters"
    reliefweb_params = {
        'limit': 1,
        'query[value]': disaster_type,
        'query[operator]': 'AND'
    }
    response = requests.get(reliefweb_url, params=reliefweb_params)
    data = response.json()

    if data.get('data'):
        fields = data['data'][0]['fields']
        disaster_name = fields.get('name', 'N/A')
        country_name = fields.get('country', [{}])[0].get('name', 'Unknown')
        location_display = location or country_name

        st.session_state.disaster_data = {
            'name': disaster_name,
            'disaster_type': disaster_type,
            'severity': severity,
            'location': location_display
        }
        st.success("✅ Disaster info fetched!")
    else:
        st.warning("No disaster info found from ReliefWeb.")

# ----------- Weather Info Input ------------
st.subheader("🌤️ Step 2: Get Weather Info")

with st.form("weather_form"):
    city_name = st.text_input("City Name (for Weather Info)")
    submitted_weather = st.form_submit_button("Submit Weather Info")

if submitted_weather:
    try:
        weather_api_key = st.secrets["weather_api_key"]
    except:
        weather_api_key = "7eb20794085a2d30dadd96937ed3e87c"

    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={weather_api_key}"
    response = requests.get(weather_url)
    weather_data = response.json()

    if 'main' in weather_data and 'weather' in weather_data:
        temperature = weather_data['main']['temp'] - 273.15
        weather_condition = weather_data['weather'][0]['description']
        lat = weather_data['coord']['lat']
        lon = weather_data['coord']['lon']

        st.session_state.weather = {
            'temperature': temperature,
            'weather_condition': weather_condition,
            'lat': lat,
            'lon': lon,
            'city': city_name
        }
        st.success("✅ Weather info fetched!")
    else:
        st.warning("Failed to fetch weather info!")

# ----------- Prediction ---------------------
st.subheader("📈 Step 3: Predict Resources")

if st.button("Predict Relief Requirements"):
    if "disaster_data" in st.session_state and "weather" in st.session_state:
        input_df = pd.DataFrame({
            'disaster_type': [st.session_state.disaster_data['disaster_type']],
            'location': [st.session_state.disaster_data['location']],
            'severity_level': [st.session_state.disaster_data['severity']],
            'weather_condition': [st.session_state.weather['weather_condition']],
            'temperature': [st.session_state.weather['temperature']],
            'road_blocked': [0],
            'population_affected': [0],
            'medical_cases': [0],
            'infrastructure_damage_percent': [0]
        })

        prediction = model.predict(input_df)

        st.session_state.prediction_result = {
            'food_required': prediction[0][0],
            'rescue_teams': prediction[0][1],
            'medical_aid': prediction[0][2]
        }

# Display prediction and map if available in session_state
if "prediction_result" in st.session_state:
    result = st.session_state.prediction_result

    st.success("🎯 Prediction Complete!")
    st.write(f"**Disaster Name:** {st.session_state.disaster_data['name']}")
    st.write(f"**Location:** {st.session_state.disaster_data['location']}")
    st.write(f"**Temperature:** {st.session_state.weather['temperature']:.2f} °C")
    st.write(f"**Weather Condition:** {st.session_state.weather['weather_condition']}")

    st.markdown("---")
    st.subheader("🔍 Prediction Results")
    st.write(f"🛒 **Food Required (Tons):** {result['food_required']:.2f}")
    st.write(f"🚑 **Rescue Teams Required:** {result['rescue_teams']:.2f}")
    st.write(f"🩺 **Medical Aid Required:** {result['medical_aid']:.2f}")

    # ------------- Gemini AI Summary ----------------
    st.subheader("🧠 AI-Generated Summary")

    report_prompt = f"""
You are an expert disaster response analyst. Based on the following inputs, generate a professional report explaining the predicted disaster relief requirements. Mention how factors like disaster type, severity, weather, and location influence the predictions.

Disaster Name: {st.session_state.disaster_data['name']}
Disaster Type: {st.session_state.disaster_data['disaster_type']}
Severity Level: {st.session_state.disaster_data['severity']}
Location: {st.session_state.disaster_data['location']}
Temperature: {st.session_state.weather['temperature']:.2f} °C
Weather Condition: {st.session_state.weather['weather_condition']}
Predicted Food (Tons): {result['food_required']:.2f}
Predicted Rescue Teams: {result['rescue_teams']:.2f}
Predicted Medical Aid Units: {result['medical_aid']:.2f}

Make it clear and media/NGO ready.
"""
    gemini_response = genai_model.generate_content(report_prompt)
    st.markdown(gemini_response.text)

    # -------- Weather Map -----------
    st.subheader("🗺️ Weather Map of Location")

    m = folium.Map(location=[st.session_state.weather['lat'], st.session_state.weather['lon']], zoom_start=8)
    popup_text = f"{st.session_state.weather['city']}<br>Temp: {st.session_state.weather['temperature']:.2f}°C<br>{st.session_state.weather['weather_condition'].title()}"
    folium.Marker(
        [st.session_state.weather['lat'], st.session_state.weather['lon']],
        popup=popup_text,
        tooltip="Weather Info",
        icon=folium.Icon(color="blue", icon="cloud")
    ).add_to(m)

    st_folium(m, width=700, height=450)
else:
    st.info("👉 Please click the prediction button to see results and map.")
