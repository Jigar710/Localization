from geopy.geocoders import Nominatim
import geocoder

# Initialize the geocoder
geolocator = Nominatim(user_agent="myGeocoder")

# Use the geocoder library to get the current location
current_location = geocoder.ip('me')

if current_location:
    latitude = current_location.latlng[0]
    longitude = current_location.latlng[1]

    print(f"Current Latitude: {latitude}")
    print(f"Current Longitude: {longitude}")
else:
    print("Failed to retrieve current location.")
