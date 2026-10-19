"""Daily surface geostrophic velocity and fluctuating kinetic energy (FKE) on a 0.25 deg grid.

FKE = 0.5 |u - <u>|^2, where <u> is a centred 90-day running mean of the geostrophic velocity.
For each dataset this writes:

    {name}_hist_GS.zarr              daily U_gs_full, V_gs_full, u_gs_eddy, v_gs_eddy, EKE (= FKE)
    EKE_mean_annual_{name}_hist.nc   annual means of FKE
    MKE_mean_annual_{name}_hist.nc   annual means of 0.5 |<u>|^2

Usage:
    python FKE_eddy_daily.py ICON FESOM NEMO AVISO_MS --out-dir OUT
    python FKE_eddy_daily.py --ssh-file 'HadGEM_HH/*.nc' --ssh-var zos --name HadGEM_HH --out-dir OUT
"""

import argparse
from pathlib import Path

import intake
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster

CATALOG = "https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml"
NEMO_FILES = (
    "/work/bm1344/DKRZ/CMOR/EERIE/HighResMIP/BSC/IFS-NEMO-ER/hist-1950/"
    "r1i1p1f1/HROday/zos/gr/v20250516/zos_HROday_*.nc"
)

TIME_SLICE = slice("1993-01-01", "2014-12-31")
T_FILTER = 90  # days
EQ_BAND = 5.0  # deg

OMEGA = 7.2921e-5  # s-1
G = 9.81  # m s-2
R_EARTH = 6371.0e3  # m


def load_icon(cat):
    ocean = cat["dkrz.disk.model-output"]["icon-esm-er"]["hist-1950"]["v20240618"]["ocean"]
    return ocean["gr025"]["2d_daily_mean"](chunks={}).to_dask().ssh


def load_fesom(cat):
    ocean = cat["dkrz.disk.model-output"]["ifs-fesom2-sr"]["hist-1950"]["v20240304"]["ocean"]
    ssh = ocean["gr025"]["2D_daily_avg_1950-2014"](chunks={}).to_dask().avg_zos
    return ssh.sel(depth=0).drop_vars("depth")


def load_nemo(cat):
    # IFS-NEMO ocean output is not in the catalogue, so read the CMOR files.
    ds = xr.open_mfdataset(
        NEMO_FILES, chunks={"time": 184, "lat": -1, "lon": -1}, combine="by_coords", parallel=True
    )
    return ds.zos


def load_aviso(cat):
    ds = cat["dkrz.disk.observations.AVISO.ssh-L4.daily_multi"](chunks={}).to_dask()
    adt = ds.adt.rename({"latitude": "lat", "longitude": "lon"})
    adt = adt.assign_coords(lon=np.mod(adt.lon, 360)).sortby("lon")
    adt = adt.coarsen(lat=2, lon=2).mean().astype(np.float32)
    return adt.chunk({"time": 180})


LOADERS = {"ICON": load_icon, "FESOM": load_fesom, "NEMO": load_nemo, "AVISO_MS": load_aviso}


def geostrophic_velocity(ssh, latlon_units="degrees"):
    # Centred differences in lat/lon units, then converted to metres.
    padded = ssh.pad(lon=(1, 1), mode="wrap").pad(lat=(1, 1), mode="wrap").chunk({"lat": -1, "lon": -1})
    deta_dlon = padded.differentiate("lon").isel(lat=slice(1, -1)).isel(lon=slice(1, -1))
    deta_dlat = padded.differentiate("lat").isel(lat=slice(1, -1)).isel(lon=slice(1, -1))

    lat = ssh.lat
    if latlon_units == "degrees":
        lat = np.deg2rad(lat)
    dx_dlon = R_EARTH * np.cos(lat)
    dy_dlat = R_EARTH
    if latlon_units == "degrees":
        dx_dlon = dx_dlon * np.pi / 180.0
        dy_dlat = dy_dlat * np.pi / 180.0

    f = 2.0 * OMEGA * np.sin(lat)
    u = -(G / f) * (deta_dlat / dy_dlat)
    v = (G / f) * (deta_dlon / dx_dlon)
    return u, v


def fke_daily(U, V):
    # Filter before cutting to TIME_SLICE, so the window can use data outside it.
    U_mean = U.rolling(time=T_FILTER, center=True).mean().sel(time=TIME_SLICE)
    V_mean = V.rolling(time=T_FILTER, center=True).mean().sel(time=TIME_SLICE)
    U = U.sel(time=TIME_SLICE)
    V = V.sel(time=TIME_SLICE)
    u_eddy = U - U_mean
    v_eddy = V - V_mean

    ds = xr.Dataset(
        {
            "U_gs_full": U,
            "V_gs_full": V,
            "u_gs_eddy": u_eddy,
            "v_gs_eddy": v_eddy,
            "EKE": 0.5 * (u_eddy**2 + v_eddy**2),
        }
    )
    ds = ds.where(np.abs(ds.lat) >= EQ_BAND)
    ds = ds.transpose("time", "lat", "lon").chunk({"time": 10, "lat": -1, "lon": -1})

    # CF time attrs left in .attrs clash with the zarr encoder.
    for key in ("units", "calendar"):
        if key in ds.time.attrs:
            ds.time.encoding[key] = ds.time.attrs.pop(key)
    return ds


def annual_means(ds):
    U_mean = ds.U_gs_full - ds.u_gs_eddy
    V_mean = ds.V_gs_full - ds.v_gs_eddy
    eke = ds.EKE.groupby("time.year").mean(dim="time").rename("EKE")
    mke = (0.5 * (U_mean**2 + V_mean**2)).groupby("time.year").mean(dim="time").rename("MKE")
    return eke, mke


def process(ssh, name, out_dir, latlon_units="degrees"):
    U, V = geostrophic_velocity(ssh, latlon_units)
    zarr_path = out_dir / f"{name}_hist_GS.zarr"
    fke_daily(U, V).to_zarr(zarr_path, mode="w")

    eke, mke = annual_means(xr.open_zarr(zarr_path))
    eke.to_netcdf(out_dir / f"EKE_mean_annual_{name}_hist.nc")
    mke.to_netcdf(out_dir / f"MKE_mean_annual_{name}_hist.nc")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help=f"any of {', '.join(LOADERS)}")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ssh-file", help="daily SSH files on regular 0.25 deg lat/lon grid")
    parser.add_argument("--ssh-var", default="zos")
    parser.add_argument("--name", help="output name for --ssh-file")
    parser.add_argument("--latlon-units", default="degrees", choices=["degrees", "radians"])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    unknown = set(args.datasets) - set(LOADERS)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    if args.ssh_file and not args.name:
        parser.error("--ssh-file needs --name")
    if not args.datasets and not args.ssh_file:
        parser.error("give at least one dataset or --ssh-file")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(LocalCluster(n_workers=args.workers, threads_per_worker=1))
    print(client.dashboard_link)

    cat = intake.open_catalog(CATALOG) if args.datasets else None
    for name in args.datasets:
        print(f"{name} ...", flush=True)
        process(LOADERS[name](cat), name, args.out_dir)

    if args.ssh_file:
        print(f"{args.name} ...", flush=True)
        ssh = xr.open_mfdataset(args.ssh_file, chunks={}, combine="by_coords", parallel=True)[args.ssh_var]
        process(ssh, args.name, args.out_dir, args.latlon_units)


if __name__ == "__main__":
    main()
