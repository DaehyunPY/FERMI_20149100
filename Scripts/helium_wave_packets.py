from httplib2 import Http

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from pandas import DataFrame
from numpy import linspace, pi
import matplotlib.pyplot as plt


# %%
flow = client.flow_from_clientsecrets(
    'client_secret_daehyun.you.tohoku.json',
    'https://www.googleapis.com/auth/spreadsheets.readonly')
store = Storage("/Users/daehyun/.credentials/"
                "com.googleapis.sheets.daehyun.you.tohoku.json")
credentials = tools.run_flow(flow, store)
http = credentials.authorize(Http())
service = discovery.build(
    'sheets', 'v4', http=http,
    discoveryServiceUrl="https://sheets.googleapis.com/$discovery/rest?version=v4")

# %%
loaded = service.spreadsheets().values().batchGet(
    spreadsheetId="1mZwYJo_gORPpPxdi5z-9P6yql0OzlUB7pZaY_nqCRYo",
    ranges="He Levels",
    valueRenderOption="UNFORMATTED_VALUE").execute()
sliced = loaded["valueRanges"][0]["values"]
df = DataFrame(sliced[1:], columns=sliced[0])

# %%
lower = -0.2
pot = linspace(lower, 0, 200)
tau = 2*pi*(-1/2/pot*27.2116)**1.5*0.02418884326505

plt.figure(figsize=(8, 8))
plt.subplot(121)
plt.plot(tau, pot)
plt.minorticks_on()
plt.grid(which='both')
yticks, _ = plt.yticks()
plt.xlim(0, 1000)
plt.ylim(lower, 0)
plt.xlabel("Kepler orbit time (fs)")
plt.ylabel("potential energy (eV)")

plt.subplot(122)
plt.barh(df['energy (eV)'], df['n'], 0.0005)
plt.minorticks_on()
plt.grid(which='both')
plt.yticks(yticks, ['']*len(yticks))
plt.xlabel("n")

plt.ylim(lower, 0)
plt.twinx()
plt.xlim(0, 20)
plt.ylim(lower+24.587, 0+24.587)
plt.ylabel("level (eV)")
plt.tight_layout()
plt.savefig("helium_wave_packets.pdf")
plt.show()
