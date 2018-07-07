#!/usr/bin/env python3
from httplib2 import Http
from itertools import combinations

from numpy import linspace, exp, gradient, log, pi, stack
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
omega_str = "24.48 eV"
fwhm_str = "90 fs"
omega = with_unit(omega_str)
sigma = with_unit(fwhm_str) / (8*log(2))**0.5  # fwhm to sigma
title = f"Photon: {omega_str}, Pulse duration: {fwhm_str}"

populations = [n ** -1.5 * (2*pi)**0.5 * sigma * exp(-((omega-lev)*sigma)**2/2)
               for _, (n, lev) in df[["n", "level"]].iterrows()]
t = linspace(t0, t1, 1000)
waves = stack([-1j * exp(1j*lev*t) * pop
               for pop, lev in zip(populations, df["level"])])
wavesqs = (waves[None, :, :]*waves[:, None, :].conj()).real
y = wavesqs.sum((0,1))
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
# plt.savefig(f"simul_wave_packets ({title}).pdf")
# plt.savefig(f"simul_wave_packets ({title}).png")
plt.show()

print("""\
Local maximums at:
    {}""".format("\n    ".join(f"({i}) {v:4.0f} fs"
                 for i, v in enumerate(as_femto_sec(t[1:][where])))))
    
# %%
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(as_femto_sec(t), y, 'k')
plt.yticks([0], [0])
plt.xlim(0, 1000)
plt.grid(True)
plt.title("Population of wave packet at a certain distance")

plt.subplot(212)
for (i, ni), (j, nj) in combinations(df[(10<df["n"])&(df["n"]<15)]["n"]
                                     .iteritems(), 2):
    plt.plot(as_femto_sec(t), wavesqs[i, j], label=f'n={ni}, n={nj}')
plt.yticks([0], [0])
plt.xlim(0, 1000)
plt.grid(True)
plt.xlabel("Time (fs)")

plt.tight_layout()
plt.figlegend()
# plt.savefig(f"simul_wave_packets_part ({title}).pdf")
# plt.savefig(f"simul_wave_packets_part ({title}).png")
plt.show()
