from httplib2 import Http

from numpy import linspace, exp, abs, gradient, log, pi
import matplotlib.pyplot as plt
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from pandas import DataFrame

from units import with_unit, as_femto_sec

# %%
store = Storage("/Users/daehyun/.credentials/"
                "com.googleapis.sheets.daehyun.you.tohoku.json")
credentials = store.get()
if not isinstance(credentials, client.OAuth2Credentials):
    flow = client.flow_from_clientsecrets(
        'client_secret_daehyun.you.tohoku.json',
        'https://www.googleapis.com/auth/spreadsheets.readonly')
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
t0, t1 = with_unit("0 fs"), with_unit("1000 fs")
omega_str = "24.44 eV"
fwhm_str = "90 fs"
omega = with_unit(omega_str)
sigma = with_unit(fwhm_str) / (8*log(2))**0.5  # fwhm to sigma
title = "Photon: {}, Pulse duration: {}".format(omega_str, fwhm_str)

populations = [n ** -1.5 * (2*pi)**0.5 * sigma * exp(-((omega-lev)*sigma)**2/2)
               for _, (n, lev) in df[["n", "level"]].iterrows()]
t = linspace(t0, t1, 1000)
waves = (-1j * exp(1j*lev*t) * pop
         for pop, lev in zip(populations, df["level"]))
y = abs(sum(waves))**2
dy = gradient(y) > 0
where = dy[:-1] & ~dy[1:]

# %%
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(df["n"], populations, 'o-')
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel("Energy level")
plt.ylabel("Coefficient")
plt.xlim(0, 40)
plt.ylim(0, None)
plt.yticks([0], [0])
plt.title(title)

plt.subplot(212)
plt.plot(as_femto_sec(t), y)
plt.plot(as_femto_sec(t[1:][where]), y[1:][where], 'o')
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel("Time (fs)")
plt.ylabel("Population of wave packet\nat a certain distance")
plt.xlim(0, 1000)
plt.ylim(0, None)
plt.yticks([0], [0])
plt.tight_layout()
plt.savefig("simul_wave_packets ({}).pdf".format(title))
plt.savefig("simul_wave_packets ({}).png".format(title))
plt.show()

print("""\
Local maximums at:
    {}""".format("\n    ".join("({}) {:4.0f} fs".format(i, v)
                 for i, v in enumerate(as_femto_sec(t[1:][where])))))
