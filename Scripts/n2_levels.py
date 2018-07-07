from httplib2 import Http

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy import arange


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
    spreadsheetId="1Xs8eHWFdb3oPMlxX6736Jx41C1qxnql902cRgEPpjWo",
    ranges="Levels",
    valueRenderOption="UNFORMATTED_VALUE").execute()
sliced = loaded["valueRanges"][0]["values"]
df = DataFrame(sliced[1:], columns=sliced[0])

# %%
plt.figure(figsize=(16, 8))
plt.bar(df["level (eV)"], df["jet absorption"]/100, 0.002)
plt.axvspan(14.6, 14.9, alpha=0.1, color='y')
plt.minorticks_on()
plt.grid(which='both')
plt.xlabel("level (eV)")
plt.ylabel("f")
plt.xticks(arange(14.4, 15.6, 0.1))
plt.tight_layout()
plt.savefig("n2_levels.pdf")
plt.show()
