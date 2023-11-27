import pywifi
from pywifi import const
import time
# Initialize Wi-Fi interface
wifi = pywifi.PyWiFi()
iface = wifi.interfaces()[0]  # Assuming the first Wi-Fi interface

# Scan for available networks
iface.scan()

# Get scan results
scan_results = iface.scan_results()
time.sleep(1)

iface.scan()
# Get scan results
scan_results2 = iface.scan_results()

scan_results.extend(scan_results2)
freq = []
wifilst = ["Jigar","DAIICT_Student", "DA_Public", "Attendance"]# "DAIICT_STAFF"
for res in scan_results:
    if(res.ssid in wifilst):
        freq.append([res.ssid, res.signal])
# Print the list of frequencies
print("Wi-Fi Frequencies:")
for frequency in freq:
    print(frequency)

finallst = []
for i in wifilst:
    for j in freq:
        if(j[0]==i):
            finallst.append(j)
            break
print(finallst)
