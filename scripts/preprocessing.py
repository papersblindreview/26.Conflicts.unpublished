"""Build the grid-cell-week dataset used by the models, from raw downloads to partition files.

Stages (each reads/writes ./data/partitions/data_final_world_XX.csv, XX = date partition 01..50):

  download   weekly climate rasters (CHIRPS precipitation, CHIRTS tmax/tmin/hi/wbgt, CAMS pm25) -> ./data/{CHIRPS,CHIRTS,CAMS}
  dataset    ACLED events on the 0.5-degree grid + biome, sustainability, EPI, population, GDP, food insecurity,
             rivers/lakes, drought, malaria and electricity covariates
  climate    merge the weekly climate rasters onto the grid; add lon/lat
  vdem       merge the V-Dem liberal democracy index

Usage (run from the repository root):
  python scripts/preprocessing.py --stage download --var all          # once; slow
  SGE_TASK_ID=<1..50> python scripts/preprocessing.py                 # dataset + climate + vdem for one partition
  python scripts/preprocessing.py --stage vdem --part 7               # a single stage for one partition
"""
import argparse
import datetime as dt
import glob
import os
import re
import sys
from functools import partial

from functions import *

BEG_YEAR, END_YEAR = 2018, 2024
PARTS = 50
CLIM_VARS = ['tmin', 'tmax', 'hi', 'wbgt', 'ppt', 'pm25']

dates = pd.date_range(start=f"{BEG_YEAR}-01-01", end=f"{END_YEAR}-12-31")
n_dates = len(dates)
partition = [dates[i * n_dates // PARTS: (i + 1) * n_dates // PARTS] for i in range(PARTS)]

_cache = {}


def part_path(p):
    return f'./data/partitions/data_final_world_{p:02d}.csv'


def get_world_grid():
    """0.5-degree grid over land (built once per process)."""
    if 'grid' not in _cache:
        world = gpd.read_file('./data/world_map/ne_10m_admin_0_countries.shp').to_crs(4326)
        _cache['grid'] = create_grid(grid_size=0.5, world=world)
    return _cache['grid']


##################################################################################################
# Stage: download
##################################################################################################

def weekly_jobs():
    """(year, week_start, week_end) for every 7-day block of every year in the study period."""
    jobs = []
    for year in range(BEG_YEAR, END_YEAR + 1):
        year_end = dt.date(year, 12, 31)
        current = dt.date(year, 1, 1)
        while current <= year_end:
            jobs.append((year, current, min(current + dt.timedelta(days=6), year_end)))
            current += dt.timedelta(days=7)
    return jobs


def _daily_tifs_to_weekly(week_start, week_end, url_fn, out_fn, var_name, out_nc, engine=None, skipna=False):
    """Download daily GeoTIFFs for one week, average them, write a compressed NetCDF, delete the dailies."""
    import requests
    import rioxarray

    daily_files = []
    for single_date in (week_start + dt.timedelta(n) for n in range((week_end - week_start).days + 1)):
        date_str = single_date.strftime("%Y.%m.%d")
        out_file = out_fn(date_str)
        r = requests.get(url_fn(date_str))
        with open(out_file, "wb") as f:
            f.write(r.content)
        daily_files.append(out_file)

    da = xr.concat([rioxarray.open_rasterio(f).squeeze() for f in daily_files], dim="time")
    da = da.where(da != -9999.)
    weekly_avg = da.mean(dim="time", skipna=True).astype("float32") if skipna else da.mean(dim="time").astype("float32")

    encoding = {var_name: {"zlib": True, "dtype": "float32"}}
    kwargs = {'engine': engine} if engine else {}
    weekly_avg.to_dataset(name=var_name).to_netcdf(out_nc(week_start.strftime("%Y.%m.%d")), encoding=encoding, **kwargs)
    for f in daily_files:
        os.remove(f)


def process_week_chirps(year, week_start, week_end):
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/"
    base_out = './data/CHIRPS/'
    _daily_tifs_to_weekly(
        week_start, week_end,
        url_fn=lambda d: f"{base_url}{year}/chirps-v3.0.rnl.{d}.tif",
        out_fn=lambda d: f'{base_out}chirps-v3.0.rnl.{d}.tif',
        var_name='ppt',
        out_nc=lambda w: f"{base_out}ppt_weekly_{w}.nc")


def process_week_chirts(year, week_start, week_end, variable):
    filename = {'tmax': 'CHIRTS-ERA5.daily_Tmax', 'tmin': 'CHIRTS-ERA5.daily_Tmin'}.get(variable, variable.upper())
    base_url = f"https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/{variable}/tifs/daily"
    base_out = f'./data/CHIRTS/{variable}'
    _daily_tifs_to_weekly(
        week_start, week_end,
        url_fn=lambda d: f"{base_url}/{d[:4]}/{filename}.{d}.tif",
        out_fn=lambda d: f'{base_out}/{variable}_{d}.tif',
        var_name=variable,
        out_nc=lambda w: f"{base_out}/{variable}_weekly_{w}.nc",
        engine='h5netcdf', skipna=True)


def process_week_pm25(year, week_start, week_end):
    """CAMS global reanalysis (EAC4) PM2.5, 12:00 each day, averaged over the week. Needs a ~/.cdsapirc key."""
    import cdsapi
    base_out = "./data/CAMS/pm25"
    c = cdsapi.Client()
    date_range = f"{week_start.strftime('%Y-%m-%d')}/{week_end.strftime('%Y-%m-%d')}"
    week_str = week_start.strftime("%Y.%m.%d")
    grib_file = f"{base_out}/pm25_{week_str}.grib"

    request = {"variable": "particulate_matter_2.5um", "model_level": "60", "date": date_range,
               "time": ["12:00"], "data_format": "grib"}
    c.retrieve("cams-global-reanalysis-eac4", request, grib_file)

    ds = xr.open_dataset(grib_file, engine="cfgrib")
    da = ds["pm2p5"]
    if "time" not in da.dims:
        da = da.expand_dims("time")
    weekly_avg = da.mean(dim="time").astype("float32")
    encoding = {"pm25": {"zlib": True, "dtype": "float32"}}
    weekly_avg.to_dataset(name="pm25").to_netcdf(f"{base_out}/pm25_weekly_{week_str}.nc", engine="h5netcdf", encoding=encoding)
    os.remove(grib_file)


def stage_download(var):
    variables = ['ppt', 'tmax', 'tmin', 'hi', 'wbgt', 'pm25'] if var == 'all' else [var]
    for v in variables:
        folder = {'ppt': './data/CHIRPS', 'pm25': './data/CAMS/pm25'}.get(v, f'./data/CHIRTS/{v}')
        os.makedirs(folder, exist_ok=True)
        if v == 'ppt':
            fn = process_week_chirps
        elif v == 'pm25':
            fn = process_week_pm25
        else:
            fn = partial(process_week_chirts, variable=v)
        Parallel(n_jobs=-1, backend="loky", verbose=10)(delayed(fn)(y, a, b) for y, a, b in weekly_jobs())


##################################################################################################
# Stage: dataset
##################################################################################################

def stage_dataset(p):
    # ---- ACLED events for this partition's dates ----
    data_all = load_all_acled(BEG_YEAR, END_YEAR)
    data = data_all[(data_all.event_date >= partition[p-1][0]) & (data_all.event_date <= partition[p-1][-1])].copy()

    # dummies for each type of event
    cat_type = pd.CategoricalDtype(categories=data_all.event_type.unique())
    data['event_type'] = data['event_type'].astype(cat_type)
    dummies = pd.get_dummies(data['event_type']).astype('int')
    dummies.columns = [re.sub(r'[^\w\s]', '', c).replace(' ', '_') for c in dummies.columns]
    concat_dummies = list(dummies.columns)
    data_dummied = pd.concat([data, dummies], axis=1)
    data_dummied.drop(columns=['disorder_type', 'event_type', 'sub_event_type'], inplace=True)

    data_political_violence = data_dummied[['event_date', 'geometry'] + concat_dummies].copy()
    data_political_violence['week_start'] = data_political_violence['event_date'].dt.to_period('W-MON').dt.start_time

    world_grid = get_world_grid()
    print('Created dummies and full world grid...\n')
    sys.stdout.flush()

    # ---- put events on the grid, weekly ----
    data_gridded = gpd.sjoin(data_political_violence, world_grid, how='left', predicate='within')
    data_gridded = data_gridded.groupby(['grid_id', 'week_start'], as_index=False)[concat_dummies].sum()
    data_gridded = merge_in_chunks(data_gridded, world_grid[['grid_id', 'geometry']], on='grid_id', how='left', parallel=False)

    # all weeks x all grid cells
    all_weeks = data_political_violence[['week_start']].drop_duplicates()
    all_grids = pd.DataFrame({'grid_id': world_grid['grid_id'].unique()})
    full_index = all_weeks.assign(key=1).merge(all_grids.assign(key=1), on='key').drop('key', axis=1)
    full_index['grid_id'] = full_index['grid_id'].astype('float')
    full_index = merge_in_chunks(full_index, world_grid, on=['grid_id'], how='left', parallel=False)
    print('Created full spatiotemporal grid...\n')
    sys.stdout.flush()

    data_full = merge_in_chunks(full_index, data_gridded.drop('geometry', axis=1), on=['grid_id', 'week_start'], how='left')
    data_full = gpd.GeoDataFrame(data_full, geometry='geometry', crs=world_grid.crs)
    numeric_cols = data_full.select_dtypes(include='number').columns
    data_full[numeric_cols] = data_full[numeric_cols].fillna(0)
    print('Created full spatiotemporal dataset...\n')
    sys.stdout.flush()

    # ---- national sustainability indicators (GHG, material footprint, ecological footprint) ----
    data_full['year'] = data_full.week_start.dt.year
    sustain_gdf = prepare_sustainability(world_grid)
    data_full = merge_in_chunks(data_full, sustain_gdf.drop('ISO_A3', axis=1), on=['grid_id', 'year'], how='left')
    data_full.drop('year', axis=1, inplace=True)

    # ---- biomes ----
    biomes_gdf = prepare_biomes(world_grid)
    data_full = merge_in_chunks(data_full, biomes_gdf[['BIOME_NAME', 'grid_id']], on='grid_id', how='left')

    # ---- EPI ----
    epi_gdf = prepare_EPI(world_grid)
    data_full['year'] = data_full.week_start.dt.year
    data_full = merge_in_chunks(data_full, epi_gdf, on=['grid_id', 'year'], how='left')
    data_full.drop(columns=['iso', 'year'], inplace=True)
    print('Loaded EPI index...\n')
    sys.stdout.flush()

    # ---- population ----
    pop_gdf = prepare_pop(world_grid)
    data_full = pd.merge(data_full, pop_gdf, on='grid_id', how='left')

    # ---- GDP, apportioned to cells by population share ----
    gdp_gdf = prepare_gdp(world_grid)
    data_full['year'] = data_full.week_start.dt.year
    data_full = merge_in_chunks(data_full, gdp_gdf, on=['grid_id', 'year'], how='left')
    data_full["country_pop"] = data_full.groupby(["ISO_A3", "year"])["population"].transform("sum")
    data_full["gdp"] = data_full["gdp"] * data_full["population"] / data_full["country_pop"]
    data_full.drop(['year', 'country_pop', 'ISO_A3'], axis=1, inplace=True)

    # ---- food insecurity (IPC phases), apportioned by population share ----
    food_gdf = prepare_food(world_grid)
    data_full['year'] = data_full.week_start.dt.year
    data_full = merge_in_chunks(data_full, food_gdf, on=['grid_id', 'year'], how='left')
    data_full["region_pop"] = data_full.groupby(["adm1_pcode", "year"])["population"].transform("sum")
    for i in range(1, 6):
        data_full[f"phase_{i}"] = data_full[f"phase_{i}"] * data_full["population"] / data_full["region_pop"]
    data_full.drop(['year', 'region_pop', 'adm1_pcode'], axis=1, inplace=True)
    data_full[[f"phase_{i}" for i in range(1, 6)]].fillna(0.0, inplace=True)
    print('Loaded population, GDP, food...\n')
    sys.stdout.flush()

    # ---- rivers (total length) and lakes (total area) per cell, in an equal-area projection ----
    data_full = data_full.to_crs(epsg=6933)
    rivers_gdf = gpd.read_file('./data/rivers/ne_10m_rivers_lake_centerlines.shp').to_crs(data_full.crs)
    river_in_cells = gpd.overlay(rivers_gdf, data_full[['grid_id', 'geometry']], how='intersection')
    river_in_cells['river_length'] = river_in_cells.geometry.length
    river_length_per_cell = river_in_cells.groupby('grid_id', as_index=False)['river_length'].sum()
    data_full = data_full.merge(river_length_per_cell, on='grid_id', how='left')

    lakes_gdf = gpd.read_file('./data/lakes/ne_10m_lakes.shp').to_crs(data_full.crs)
    lakes_in_cells = gpd.overlay(lakes_gdf, data_full[['grid_id', 'geometry']], how='intersection')
    lakes_in_cells['lake_area'] = lakes_in_cells.geometry.area
    lake_area_per_cell = lakes_in_cells.groupby('grid_id', as_index=False)['lake_area'].sum()
    data_full = data_full.merge(lake_area_per_cell, on='grid_id', how='left')

    data_full['river_length'] = data_full['river_length'].fillna(0.0)
    data_full['lake_area'] = data_full['lake_area'].fillna(0.0)
    data_full = data_full.to_crs(4326)
    print('Loaded rivers and lakes...\n')
    sys.stdout.flush()

    # ---- drought ----
    drought_gdf = prepare_drought(world_grid)
    data_full = merge_in_chunks(data_full, drought_gdf, on=['grid_id', 'week_start'], how='left')
    data_full['drought'] = data_full['drought'].fillna(0.0)
    print('Loaded drought...\n')
    sys.stdout.flush()

    # ---- malaria ----
    malaria_gdf = prepare_malaria(world_grid)
    data_full['year'] = data_full.week_start.dt.year
    data_full = merge_in_chunks(data_full, malaria_gdf, on=['grid_id', 'year'], how='left')
    data_full.drop('year', axis=1, inplace=True)
    data_full['malaria'] = data_full['malaria'].fillna(0.0)

    # ---- electricity access (national share of population) ----
    elect_gdf = prepare_electricity(world_grid)
    data_full = merge_in_chunks(data_full, elect_gdf, on=['grid_id'], how='left')

    os.makedirs('./data/partitions', exist_ok=True)
    data_full.to_csv(part_path(p), index=False)


##################################################################################################
# Stage: climate
##################################################################################################

def stage_climate(p):
    world_grid = get_world_grid()
    start_date = partition[p-1][0].to_pydatetime()
    end_date = partition[p-1][-1].to_pydatetime()

    data_full = pd.read_csv(part_path(p))
    data_full.drop('pm25', axis=1, inplace=True, errors='ignore')   # pm25 is re-merged below
    data_full['week_start'] = pd.to_datetime(data_full['week_start'])

    gdfs = []
    for v in CLIM_VARS:
        if v == 'ppt':
            folder = './data/CHIRPS/'
        elif v == 'pm25':
            folder = f'./data/CAMS/{v}/'
        else:
            folder = f'./data/CHIRTS/{v}/'
        all_files = sorted(glob.glob(os.path.join(folder, '*.nc')))

        # only keep climate files in the date range
        files = [f for f in all_files if start_date <= extract_date(f) <= end_date]

        gdfs_var = [process_clim_file(f, v, world_grid, data_full) for f in files]
        gdfs_var = [g for g in gdfs_var if g is not None]

        combined_gdf = concat_in_chunks(gdfs_var)   # in chunks to save memory
        gdfs.append(combined_gdf)
        print(f'Processed {v}...\n')
        sys.stdout.flush()

    data_final = gpd.GeoDataFrame(gdfs[0], geometry='geometry', crs="EPSG:4326")
    for g, v in zip(gdfs[1:], CLIM_VARS[1:]):
        data_final[v] = g[v]

    print(data_final.isna().sum())
    print(data_final[['pm25']].describe())

    # cell centroids as lon/lat
    data_final = data_final.to_crs(3857)
    data_final['geometry'] = data_final.geometry.centroid
    data_final = data_final.to_crs(4326)
    data_final['lon'] = data_final.geometry.x
    data_final['lat'] = data_final.geometry.y
    data_final.drop(columns=['geometry'], inplace=True)

    num_cols = data_final.select_dtypes(include='number').columns
    data_final[num_cols] = data_final[num_cols].astype('float32')

    print(data_final.shape)
    print(data_final.isna().sum() / data_final.shape[0])

    data_final.dropna(subset=['lon', 'lat'], inplace=True)
    data_final.to_csv(part_path(p), index=False)


##################################################################################################
# Stage: vdem
##################################################################################################

def stage_vdem(p):
    world_grid = get_world_grid()
    data_final = pd.read_csv(part_path(p))
    data_final['week_start'] = pd.to_datetime(data_final.week_start)
    data_final['year'] = data_final.week_start.dt.year

    vdem_gdf = prepare_vdem(world_grid)
    data_final = merge_in_chunks(data_final, vdem_gdf.drop('ISO_A3', axis=1), on=['grid_id', 'year'], how='left')
    data_final.drop('year', axis=1, inplace=True)
    data_final.to_csv(part_path(p), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['download', 'dataset', 'climate', 'vdem', 'all'], default='all',
                        help="'all' runs dataset, climate and vdem (not download)")
    parser.add_argument('--var', choices=['all', 'ppt', 'tmax', 'tmin', 'hi', 'wbgt', 'pm25'], default='all',
                        help='download stage only')
    parser.add_argument('--part', type=int, default=None, help='date partition 1..50 (default: SGE_TASK_ID, else 1)')
    args = parser.parse_args()

    if args.part is not None:
        p = args.part
    else:
        try:
            p = int(os.getenv('SGE_TASK_ID'))
        except (TypeError, ValueError):
            p = 1

    if args.stage == 'download':
        stage_download(args.var)
    else:
        if args.stage in ('dataset', 'all'):
            stage_dataset(p)
        if args.stage in ('climate', 'all'):
            stage_climate(p)
        if args.stage in ('vdem', 'all'):
            stage_vdem(p)
