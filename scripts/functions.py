"""Shared helpers: ACLED loading, 0.5-degree grid, covariate preparation, feature engineering."""

from __future__ import annotations
import os
    
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
import zipfile
import rasterio
import xarray as xr
import re
import sys
import gc
import shap
from functools import reduce
from joblib import Parallel, delayed
from datetime import datetime

def load_all_acled(beg, end):
  acled_dir = './data/acled'
  acled_files = os.listdir(acled_dir)
  data_list = []
  for d in acled_files:
    data_temp = pd.read_csv(f'{acled_dir}/{d}', low_memory=False)
    data_temp.drop_duplicates(inplace=True)
    data_list.append(data_temp)
  data = pd.concat(data_list, axis=0, ignore_index=True)
  
  data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude), crs=4326)
  
  data.drop(columns=['source','source_scale','notes','actor1','assoc_actor_1','inter1','actor2','assoc_actor_2','inter2'], inplace=True)
  data.drop(columns=['iso','region','tags','timestamp','country','year'], inplace=True) #
  data.drop(columns=['admin1','admin2','admin3','location','geo_precision','time_precision','event_id_cnty'], inplace=True)
  data.drop(columns=['interaction','civilian_targeting','fatalities','population_best'], inplace=True)
  data['event_date'] = pd.to_datetime(data['event_date'])
  data.sort_values(by='event_date', inplace=True)
  
  return data[(data.event_date.dt.year >= beg) & (data.event_date.dt.year <= end)]

def extract_date(filename):
  date_str = filename.split('_')[-1].replace('.nc', '')
  return datetime.strptime(date_str, "%Y.%m.%d")



def create_grid(grid_size, world):
  minx, miny, maxx, maxy = world.total_bounds
  
  xs = np.arange(minx, maxx, grid_size)
  ys = np.arange(miny, maxy, grid_size)
  xx, yy = np.meshgrid(xs, ys, indexing='ij')
  xy = np.stack((xx.ravel(), yy.ravel()))

  polygons = []
  for i in range(xy.shape[1]):
    x, y = xy[0,i], xy[1,i]
    polygons.append(Polygon([(x, y), (x+grid_size, y), (x+grid_size, y+grid_size), (x, y+grid_size)]))
  
  world_grid = gpd.GeoDataFrame({'geometry': polygons}, crs=world.crs)
  world_union = world.geometry.unary_union
  world_union_gpd = gpd.GeoDataFrame(geometry=[world_union], crs=world.crs)
  
  chunk_size = 5000
  indices = np.arange(0, len(world_grid), chunk_size)
  def process_gpd_chunk(i, world_grid=world_grid, world_union_gpd=world_union_gpd):
    return gpd.sjoin(world_grid.iloc[i:i+chunk_size], world_union_gpd, predicate="intersects", how="inner").drop(columns="index_right")
    
  chunks = Parallel(n_jobs=-1, verbose=10, backend="loky")(delayed(process_gpd_chunk)(i) for i in indices)
  
  world_grid = pd.concat(chunks, axis=0, ignore_index=True)
  world_grid = gpd.GeoDataFrame(world_grid, geometry='geometry', crs=world.crs)
  world_grid['grid_id'] = range(len(world_grid))
  
  return world_grid

    
def prepare_elevation(world_grid):    
  raster_path = "./data/elevation/extracted_lpk/w001001.adf"
  
  src = rasterio.open(raster_path)
  data = src.read(1) 
  nodata = src.nodata
  transform = src.transform
  
  # Downsample by taking every N-th row/col
  step = 10
  rows = np.arange(0, data.shape[0], step)
  cols = np.arange(0, data.shape[1], step)
  
  row_idx, col_idx = np.meshgrid(rows, cols, indexing="ij")
  row_idx = row_idx.flatten()
  col_idx = col_idx.flatten()
  
  # Filter out nodata
  mask = (data[row_idx, col_idx] != nodata) if nodata is not None else np.ones_like(row_idx, dtype=bool)
  row_idx = row_idx[mask]
  col_idx = col_idx[mask]
  
  # Convert to coordinates
  xs, ys = rasterio.transform.xy(transform, row_idx, col_idx, offset='center')
  elev = data[row_idx, col_idx]
  
  # Build GeoDataFrame
  gdf = gpd.GeoDataFrame({"elevation": elev}, geometry=gpd.points_from_xy(xs, ys), crs=4326)
  gdf = gpd.clip(gdf, world_grid)
  
  elev_gridded = gpd.sjoin(gdf, world_grid, how='left', predicate='within')
  return elev_gridded.groupby(['grid_id'], as_index=False)['elevation'].mean()

  
def prepare_pop(world_grid):
  
  with rasterio.open("./data/population/GHS_POP_E2010_GLOBE_R2023A_4326_30ss_V1_0.tif") as src:
    data = src.read(1)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    
  step = 10
  rows = np.arange(0, data.shape[0], step)
  cols = np.arange(0, data.shape[1], step)
  
  row_idx, col_idx = np.meshgrid(rows, cols, indexing="ij")
  row_idx = row_idx.flatten()
  col_idx = col_idx.flatten()
  
  # Filter out nodata
  mask = (data[row_idx, col_idx] != nodata) if nodata is not None else np.ones_like(row_idx, dtype=bool)
  row_idx = row_idx[mask]
  col_idx = col_idx[mask]
  
  # Coordinates
  xs, ys = rasterio.transform.xy(transform, row_idx, col_idx, offset='center')
  values = data[row_idx, col_idx]
  
  gdf = gpd.GeoDataFrame({"population": values}, geometry=gpd.points_from_xy(xs, ys), crs=crs)
  gdf = gpd.clip(gdf, world_grid)
  
  pop_gridded = gpd.sjoin(gdf, world_grid, how='left', predicate='within')
  return pop_gridded.groupby(['grid_id'], as_index=False)['population'].sum()


def process_drought_file(filename, world_grid):
  
  date_str = filename.split('/')[-1].split('_')[3]
  date = f'{date_str[:4]}/{date_str[4:6]}/{date_str[-2:]}'

  with rasterio.open(filename) as src:
    data = src.read(1)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    
  rows = np.arange(0, data.shape[0])
  cols = np.arange(0, data.shape[1])
  
  row_idx, col_idx = np.meshgrid(rows, cols, indexing="ij")
  row_idx = row_idx.flatten()
  col_idx = col_idx.flatten()
  
  # Filter out nodata
  mask = (data[row_idx, col_idx] != nodata) if nodata is not None else np.ones_like(row_idx, dtype=bool)
  row_idx = row_idx[mask]
  col_idx = col_idx[mask]
  
  # Coordinates
  xs, ys = rasterio.transform.xy(transform, row_idx, col_idx, offset='center')
  values = data[row_idx, col_idx]
  
  gdf = gpd.GeoDataFrame({"drought": values}, geometry=gpd.points_from_xy(xs, ys), crs=crs)
  gdf = gpd.clip(gdf, world_grid)

  gdf_gridded = gpd.sjoin(gdf, world_grid, how='left', predicate='within')
  gdf_gridded = gdf_gridded.groupby(['grid_id'], as_index=False)['drought'].mean()
  gdf_gridded['date'] = date
  gdf_gridded['week_start'] = pd.to_datetime(gdf_gridded['date'])
  gdf_gridded['week_start'] = gdf_gridded['week_start'].dt.to_period('W-MON').dt.start_time
  return gdf_gridded.fillna(0.).drop(columns='date')

def get_drought_year(year, world_grid):
  
  folder = f'./data/drought/{year}/'
  all_files = [f for f in os.listdir(folder) if str(year) in f]
  dfs = Parallel(n_jobs=-1, verbose=0, backend="loky")(delayed(process_drought_file)(folder+f, world_grid) for f in all_files)
  
  return pd.concat(dfs, ignore_index=True)

def prepare_drought(world_grid):
  
  dfs = []
  for year in range(2018, 2025):
    dfs.append(get_drought_year(year, world_grid))
  return pd.concat(dfs, ignore_index=True)
  
def merge_in_chunks(df1, df2, on, how, parallel=True):
  chunk_size = 50000
  def process_pd_chunk(i, df1=df1, df2=df2, on=on, how=how):
      return pd.merge(df1.iloc[i:i+chunk_size], df2, on=on, how=how)
      
  indices = np.arange(0, len(df1), chunk_size)
  if parallel:      
    chunks = Parallel(n_jobs=-1, verbose=10, backend="loky")(delayed(process_pd_chunk)(i) for i in indices)
  else:
    chunks = []
    for i in indices:
      chunks.append(process_pd_chunk(i, df1=df1, df2=df2, on=on, how=how))
      
  return pd.concat(chunks, ignore_index=True)
  
def prepare_malaria(world_grid):
  files = os.listdir("./data/socioeconomic/malaria/")
  files = os.listdir("./data/socioeconomic/malaria/")
  gdfs = []
  for fi in files:
    with rasterio.open(f"./data/socioeconomic/malaria/{fi}") as src:
      data = src.read(1)
      transform = src.transform
      crs = src.crs
      
    rows, cols = np.where(data != -9999.)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    gdf = gpd.GeoDataFrame({"malaria": data[rows, cols]}, geometry=gpd.GeoSeries.from_xy(xs, ys), crs=crs)
    gdf_gridded = gpd.sjoin(gdf, world_grid, how='left', predicate='within')
    gdf_gridded = gdf_gridded.groupby(['grid_id'], as_index=False)['malaria'].mean()
    gdf_gridded['year'] = int(fi.split('_')[-1][:4])
    gdfs.append(gdf_gridded)
  
  return pd.concat(gdfs, ignore_index=True)
 
def get_world_bank_map():
  wb2 = gpd.read_file('./data/world_bank/WB_GAD_ADM2.shp')
  wb2.sort_values(by=['ISO_A3'], inplace=True)
  wb2 = wb2[['ISO_A3','geometry']]
  return wb2.dissolve(by=['ISO_A3']).reset_index()
  
def prepare_EPI(world_grid):
  wb2 = get_world_bank_map()
  wb2.rename(columns={'ISO_A3':'iso'}, inplace=True)
  
  epi_weights = pd.read_csv('./data/socioeconomic/EPI/epi2024weights.csv')
  epi_weights = epi_weights[epi_weights.Type == 'Indicator']
  epi_weights = epi_weights[['Abbreviation','Weight']].copy()
  epi_weights.sort_values(by='Abbreviation', inplace=True)

  epi_files = [f for f in os.listdir('./data/socioeconomic/EPI') if 'ind_na' in f]
  dfs = []
  for epi_file in epi_files:
      df_raw = pd.read_csv('./data/socioeconomic/EPI/' + epi_file)
      ind = epi_file.split('_')[0]
      cols_to_keep = ['iso'] + [c for c in df_raw.columns if ind in c]
      df_raw = df_raw[cols_to_keep].copy()
  
      df = pd.melt(df_raw, id_vars='iso', var_name=ind+'_raw', value_name=ind)
      df['year'] = df[ind+'_raw'].apply(lambda x: int(x.split('.')[-1]))
      dfs.append(df[['iso','year',ind]])
  
  epi = reduce(lambda left, right: pd.merge(left, right, on=["iso","year"], how="outer"), dfs)
  epi = epi[epi.year >= 2018]
  
  abbrevs = [c for c in epi.columns if c in epi_weights.Abbreviation.unique()] 
  epi = epi[['iso','year'] + abbrevs]
  
  weights = epi_weights.Weight.values.reshape(1,-1)
  epi[abbrevs] = epi[abbrevs] * weights
  
  epi_index = epi[abbrevs].sum(axis=1) / weights.sum().sum()
  epi.drop(columns=abbrevs, inplace=True)
  epi['EPI_index'] = epi_index
  
  epi = pd.merge(epi, wb2, on='iso', how='left')
  gdp_merged = gpd.GeoDataFrame(epi, geometry='geometry', crs=4326)
  
  gdf = gpd.GeoDataFrame(gdp_merged, geometry='geometry', crs=world_grid.crs)
  gdf_gridded = gpd.sjoin(gdf, world_grid, how='left', predicate='intersects')
  return gdf_gridded.groupby(['iso','grid_id','year'], as_index=False)['EPI_index'].mean()
 
def prepare_sustainability(world_grid):
  wb2 = get_world_bank_map()
  
  ghg = pd.read_csv('./data/socioeconomic/GHG_per_capita_t_1970_2024.csv')
  mf = pd.read_csv('./data/socioeconomic/MF_per_capita_t_1970_2024.csv')
  ef = pd.read_csv('./data/socioeconomic/ECOLOGICAL_FOOTPRINT_per_capita_1970_2024.csv')
  
  sustain_df = pd.merge(ghg, mf, on=['Country','ISO3','YEAR'])
  sustain_df = pd.merge(sustain_df, ef, on=['Country','ISO3','YEAR'])
  sustain_df.drop('Country', axis=1, inplace=True)
  sustain_df.rename(columns={'YEAR':'year','ISO3':'ISO_A3'}, inplace=True)
  
  sustain_df = sustain_df[(sustain_df.year >= 2018) & (sustain_df.year <= 2024)]
  
  sustain_merged = pd.merge(sustain_df, wb2, on=['ISO_A3'], how='left')
  
  sustain = gpd.GeoDataFrame(sustain_merged, geometry='geometry', crs=world_grid.crs)
  sustain_gridded = gpd.sjoin(sustain, world_grid, how='left', predicate='intersects')
  return sustain_gridded.groupby(['ISO_A3','grid_id','year'], as_index=False)[['GHG_pc_t', 'MF_pc_t', 'ECO_pc_gha']].mean()
   

def prepare_biomes(world_grid): 
  biomes = gpd.read_file('./data/biomes/Ecoregions2017.shp')
  biomes = biomes.to_crs(world_grid.crs)
  
  centroids = world_grid.copy()
  centroids["geometry"] = centroids.geometry.centroid
    
  joined = gpd.sjoin(centroids, biomes[["BIOME_NAME", "geometry"]], how="left", predicate="within")
  
  joined["geometry"] = world_grid.geometry
  return joined
  
  
def prepare_gdp(world_grid):
  wb2 = get_world_bank_map()
  
  gdp = pd.read_csv('./data/socioeconomic/GDP.csv')
  gdp.drop(columns=['Indicator Name','Indicator Code'], inplace=True)
  gdp = pd.melt(gdp, id_vars='Country Code', value_vars=gdp.columns[2:], var_name='year', value_name='gdp')
  gdp['year'] = gdp['year'].astype(int)
  gdp = gdp[(gdp.year >= 2018) & (gdp.year <= 2024)]
  
  gdp.rename(columns={'Country Code':'ISO_A3'}, inplace=True)
  gdp_merged = pd.merge(gdp, wb2, on=['ISO_A3'], how='left')
  
  gdf = gpd.GeoDataFrame(gdp_merged, geometry='geometry', crs=world_grid.crs)
  gdf_gridded = gpd.sjoin(gdf, world_grid, how='left', predicate='intersects')
  return gdf_gridded.groupby(['ISO_A3','grid_id','year'], as_index=False)['gdp'].mean()
  

def prepare_food(world_grid):

  geom = gpd.read_file("./data/socioeconomic/global_admin_boundaries_matched_latest.gdb", layer="admin1")
  geom = geom[['iso3','adm1_pcode','geometry']].copy()
  geom.drop_duplicates(subset=['iso3','adm1_pcode'], inplace=True)

  food = pd.read_csv('./data/socioeconomic/hdx_hapi_food_security_global.csv', low_memory=False)
  
  food = food[food.ipc_type == 'current']
  food.drop(columns=['dataset_hdx_id','resource_hdx_id','warning','error','ipc_type','has_hrp','in_gho'], inplace=True)
  food.drop(columns=['provider_admin1_name','provider_admin2_name','admin2_code','admin2_name'], inplace=True)
  
  food1 = food[food.admin_level == 1].copy()
  food1.drop('admin_level', axis=1, inplace=True)
  food1 = food1[~food1.ipc_phase.isin(['all','3+'])]
  
  unique_id = ["location_code",'admin1_code','reference_period_start','reference_period_end']
  food1_wide = food1.pivot_table(index=unique_id, columns="ipc_phase", values='population_in_phase').reset_index()
  food1_wide.sort_values(by=['location_code','admin1_code','reference_period_start'], inplace=True)
  food1_wide.rename(columns={'location_code':'iso3','admin1_code':'adm1_pcode'}, inplace=True)
  
  food_merged = pd.merge(food1_wide, geom, on=['iso3','adm1_pcode'], how="left")
  food_merged = gpd.GeoDataFrame(food_merged, geometry='geometry', crs=world_grid.crs)
  food_merged['reference_period_start'] = pd.to_datetime(food_merged['reference_period_start'])
  food_merged['reference_period_end'] = pd.to_datetime(food_merged['reference_period_end'])
  food_merged['year'] = food_merged[['reference_period_end','reference_period_start']].mean(axis=1).dt.year
  food_merged.drop(columns=['reference_period_end','reference_period_start','iso3'], inplace=True)
  food_merged.columns = ['adm1_pcode'] + [f'phase_{i}' for i in range(1,6)] + list(food_merged.columns[-2:])
  
  gdf_gridded = gpd.sjoin(food_merged, world_grid, how='left', predicate='intersects')
  return gdf_gridded.groupby(['adm1_pcode','grid_id','year'], as_index=False)[[f'phase_{i}' for i in range(1,6)]].sum()

def prepare_vdem(world_grid):
  wb2 = get_world_bank_map()

  vdem = pd.read_csv('./data/socioeconomic/vdem_libdem_index.csv')
  vdem = pd.merge(vdem, wb2, on=['ISO_A3'], how='left')
  gdf = gpd.GeoDataFrame(vdem, geometry='geometry', crs=world_grid.crs)

  gdf_gridded = gpd.sjoin(gdf, world_grid, how='left', predicate='intersects')
  return gdf_gridded.groupby(['ISO_A3','grid_id','year'], as_index=False)['v2x_libdem'].mean()
  

def prepare_electricity(world_grid):
  wb2 = get_world_bank_map()
  
  electr = pd.read_csv('./data/socioeconomic/electricity.csv')
  numeric_cols = electr.columns[5:]
  electr[electr[numeric_cols]<0] = np.nan
  electr['electricity_access'] = electr.Pop_A2 * (electr.SDG711pct / 100)
  electr.rename(columns={'ISO3':'ISO_A3'}, inplace=True)
  
  electr_agg = electr.groupby(['ISO_A3'], as_index=False)[['Pop_A2','electricity_access']].sum()
  electr_agg['electricity_access'] = electr_agg['electricity_access'] / electr_agg['Pop_A2']
  electr_agg['electricity_access'] = electr_agg['electricity_access'].replace(np.nan, 0)
  electr_agg.drop('Pop_A2', axis=1, inplace=True)
  electr_merged = pd.merge(electr_agg, wb2, on=['ISO_A3'], how='left')
  
  gdf = gpd.GeoDataFrame(electr_merged, geometry='geometry', crs=world_grid.crs)
  gdf_gridded = gpd.sjoin(gdf.drop('ISO_A3', axis=1), world_grid, how='left', predicate='intersects')
  return gdf_gridded.groupby(['grid_id'], as_index=False)['electricity_access'].mean()
  
def concat_in_chunks(dfs):
  
  chunks = []
  chunk_size = 100
  for i in range(0, len(dfs), chunk_size):
    combined = pd.concat(dfs[i:i+chunk_size], ignore_index=True)
    chunks.append(combined)
    del combined
    gc.collect()
  return pd.concat(chunks, ignore_index=True)
  


def process_clim_file(fpath, var, world_grid, data_full):
  ds = xr.open_dataset(fpath)
  da = ds[var]
  
  # Convert to DataFrame
  df = da.to_dataframe(name=var).reset_index()
  
  # Convert to GeoDataFrame
  
  if 'CAMS' in fpath:
    df['longitude'] = df['longitude'] - 180.
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs=4326)
  else:
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs=4326)
  gdf[var] = gdf[var].astype(float)
  gdf.loc[gdf[var] == (-9999.), var] = np.nan
  
  # Clip 
  gdf = gpd.clip(gdf, world_grid)
  
  date_match = re.search(r'(\d{4}\.\d{2}\.\d{2})', os.path.basename(fpath))
  gdf['date'] = pd.to_datetime(date_match.group(1), format='%Y.%m.%d')
  gdf['week_start'] = gdf['date'].dt.to_period('W-MON').dt.start_time
  
  data_temp = data_full[data_full.week_start.isin(gdf.week_start.unique())]
  
  if 'CAMS' in fpath:
    gdf = gpd.sjoin(gdf, world_grid, how='left', predicate='intersects')
  else:
    gdf = gpd.sjoin(gdf, world_grid, how='left', predicate='within')
    
  gdf_mean = gdf.groupby(['week_start','grid_id'], as_index=False)[var].mean()
  
  data_temp = merge_in_chunks(data_temp, gdf_mean, on=['week_start','grid_id'], how='left', parallel=False)
  data_temp = data_temp.drop('geometry', axis=1)
  data_temp = merge_in_chunks(data_temp, world_grid[['grid_id', 'geometry']], on='grid_id', how='left', parallel=False)
  ds.close()
  return data_temp
   

def feature_engineering(data, climate_vars, count_lags, count_rolling, clim_lags, clim_rolling):
  data['cell_id'] = data['lon'].astype(str) + '_' + data['lat'].astype(str)
  data = data.sort_values(by=['cell_id', 'week_start'])
  y_names = ['Battles', 'ExplosionsRemote_violence', 'Violence_against_civilians','Riots','Protests']
  
  for lag in count_lags:
    new_cols = [f'{y}_lag_{lag}' for y in y_names]
    data[new_cols] = data.groupby('cell_id')[y_names].shift(lag)
  
  for window in count_rolling:
    new_cols = [f'{y}_rollsum_{window}' for y in y_names]
    data[new_cols] = data.groupby('cell_id')[y_names].shift(1).rolling(window).sum()
  
  for lag in clim_lags:
    new_cols = [f'{var}_lag_{lag}' for var in climate_vars] 
    data[new_cols] = data.groupby('cell_id')[climate_vars].shift(lag)
  
  for window in clim_rolling:
    new_cols = [f'{var}_rollmean_{window}' for var in climate_vars] 
    data[new_cols] = data.groupby('cell_id')[climate_vars].shift(1).rolling(window).mean()
      
  return data


def prepare_data(continent, climate_vars, count_lags, count_rolling, clim_lags, clim_rolling):
  world = gpd.read_file('./data/world_map/ne_10m_admin_0_countries.shp')
  continent_groups = {"europe": ["Europe"],  'africa': ['Africa'],
                      "asia": ["Asia", "Oceania"],
                      "americas": ["North America", "South America"]}
  
  base_path = './data/partitions/'
  all_parts = [f for f in os.listdir(base_path) if 'csv' in f]
  def load_csv(file_name):
    return pd.read_csv(os.path.join(base_path, file_name))

  results = Parallel(n_jobs=-1)(delayed(load_csv)(f) for f in all_parts)
  data = pd.concat(results, ignore_index=True)
  
  data['week_start'] = pd.to_datetime(data['week_start'])
  data.sort_values(by='week_start', inplace=True)
  _, week_indices = np.unique(data['week_start'], return_inverse=True)
  data['t'] = week_indices
  
  if continent != 'world':
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["lon"], data["lat"]), crs=4326)
    world = world.to_crs(gdf.crs)
    world_sub = world[world["CONTINENT"].isin(continent_groups[continent])]
    data = gpd.sjoin(gdf, world_sub[["geometry"]], how="inner", predicate="within").drop(columns=['index_right','geometry'])
    
  
  y_names = [['Battles', 'ExplosionsRemote_violence', 'Violence_against_civilians'],['Riots','Protests']]
  y_groups = ['Conflict','Unrest']
  for i in range(2):
    output_vars = y_names[i]
    output_var = y_groups[i]
    data[output_var] = (data[output_vars].sum(axis=1) > 0).astype(int)

  data = feature_engineering(data, climate_vars, count_lags, count_rolling, clim_lags, clim_rolling)
  data.drop(columns=y_names[0]+y_names[1], inplace=True)
  
  covariate_names = list(data.drop(columns=['cell_id','grid_id','week_start','lon','lat','t','BIOME_NAME'] + y_groups + ['Strategic_developments']).columns)
  data = data.replace([np.inf, -np.inf], np.nan)
  return data, covariate_names

def read_model_configs(filename="model_configs.txt"):
  configs = []
  with open(filename, 'r') as f:
    for line in f:
      line = line.strip()
      config = {}
      parts = line.split(';')
      for part in parts:
        key, val = part.split('=')
        key = key.strip()
        val = val.strip()
        config[key] = eval(val) if val else []
      configs.append(config)
  return configs



def centroid_to_cell(centroid, grid_size=0.5):
  x, y = centroid.x, centroid.y
  return box(x - grid_size/2, y - grid_size/2, x + grid_size/2, y + grid_size/2)
   

def get_world(continent): 
  world = gpd.read_file('./data/world_map/ne_10m_admin_0_countries.shp')
  world = world.to_crs(4326)
  
  if continent != 'world':
    continent_groups = {"europe": ["Europe"], 'africa': ['Africa'],
                        "asia": ["Asia", "Oceania"],
                        "americas": ["North America", "South America"]}
    
    return world[world["CONTINENT"].isin(continent_groups[continent])].dissolve()
      
  return world.dissolve()
  
  
def plot_preds(data, week_to_plot, mod_type, continent, output_var, line):
  risk_cmap = LinearSegmentedColormap.from_list("risk", ["#ffffff", "#fff7bc", "#fec44f", "#f03b20", "#7f0000"], N=256)
  world = get_world(continent)
  
  week_to_plot_fmt = pd.to_datetime(week_to_plot).strftime('%Y/%m/%d')
  week_tag = pd.to_datetime(week_to_plot).strftime('%Y.%m.%d')
  
  fig, axes = plt.subplots(1, 2, figsize=(15,6), sharex=True, sharey=True, constrained_layout=True)

  cmap = 'turbo' #risk_cmap #
    
  df_week = data[data.week_start == week_to_plot]
  df_week.to_csv(f'./results/preds_{output_var}.csv', index=False)
  
  gdf_week = gpd.GeoDataFrame(df_week, geometry=gpd.points_from_xy(df_week.lon, df_week.lat), crs=4326)
  gdf_week[output_var] = gdf_week[output_var].replace(0, np.nan)
  gdf_week['geometry'] = gdf_week.geometry.apply(lambda p: centroid_to_cell(p))
  gdf_week = gdf_week.set_geometry('geometry')
  
  # Row 1: observed
  gdf_week.plot(ax=axes[0], column=output_var, cmap=cmap, vmin=0, vmax=1, missing_kwds={"color": "lightgrey"}, legend=False)
  world.boundary.plot(ax=axes[0], color='black')
  axes[0].set_title(f"{week_to_plot_fmt} - Observed")

  # Row 2: predicted counts
  gdf_week.plot(ax=axes[1], column=f"p_mean_{output_var}", cmap=cmap, vmin=0, vmax=1, missing_kwds={"color": "lightgrey"}, legend=False)
  world.boundary.plot(ax=axes[1], color='black')
  axes[1].set_title(f"{week_to_plot_fmt} - Predicted")
  
  for ax in axes.ravel():
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if continent == 'world':
      ax.set_xlim(-139, 159)
      ax.set_ylim(-59, 59)
    elif continent == 'asia':
      ax.set_xlim(10, 159)
      ax.set_ylim(-59, 59)
    elif continent == 'europe':
      ax.set_xlim(-10, 55)
      ax.set_ylim(29, 59)
    elif continent == 'americas':
      ax.set_xlim(-139,-25)
      ax.set_ylim(-59, 59)

  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
  sm._A = []
  cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.4, pad=0.02)
  cbar.set_label('Predicted probability', fontsize=15)
  cbar.ax.tick_params(labelsize=12)
  
  plt.savefig(f'./figures/map_{mod_type}_{continent}_{line:02d}_{output_var}_{week_tag}.png', dpi=300)
  plt.close(fig)

def get_season(month):
    if month in [12,1,2]:
      return 'Winter'
    elif month in [3,4,5]:
      return 'Spring'
    elif month in [6,7,8]:
      return 'Summer'
    else:
      return 'Fall'
'''
def plot_spatial_shap(c, shap_df, mod_type, continent, line):
  world = get_world(continent)
  
  fname = c.replace(" ", "_").replace("/", "_") 
  fig, axes = plt.subplots(2, 2, figsize=(30,15), layout='constrained')
  for i, output_var in enumerate(["Neither", "Conflict", "Unrest", "Both"]):
    shap_df_temp = shap_df_reg.groupby(['lon','lat'], as_index=False)[c].mean()
    shap_df_temp = gpd.GeoDataFrame(shap_df_temp, geometry=gpd.points_from_xy(shap_df_temp.lon, shap_df_temp.lat), crs=4326)
    
    vmax_shap = np.percentile(np.abs(shap_df_temp[c].values), 99)
  
    ax = axes[i//2,i%2]
    
    world.boundary.plot(ax=ax, color='black')
    shap_df_temp.plot(ax=ax, column=c, cmap='coolwarm', vmin=-vmax_shap, vmax=vmax_shap, legend=False, marker='s', edgecolor='none', markersize=5)
    ax.set_xlabel("Longitude", fontsize=15)
    ax.set_ylabel("Latitude", fontsize=15)
    ax.set_xlim(-139, 159)
    ax.set_ylim(-59, 59)
    
  sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-vmax_shap, vmax=vmax_shap))
  sm._A = []
  cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.5, pad=0.02)
  cbar.set_label("SHAP value (log expected probability)", fontsize=15)
  cbar.ax.tick_params(labelsize=12)
    
  plt.savefig(f"./figures/shap_space_{line:02d}_{fname}.png", dpi=300)
  plt.close(fig)
'''

def plot_multi_preds(data, week_to_plot, outputs):
  world = get_world('world')
  
  week_to_plot_fmt = pd.to_datetime(week_to_plot).strftime('%Y/%m/%d')
  week_tag = pd.to_datetime(week_to_plot).strftime('%Y.%m.%d')
  
  for output_var in outputs:
    data.loc[:,output_var] = (data.loc[:,'multi_output_str'] == output_var).astype(int)

  data[outputs] = data[outputs].replace(0, np.nan)
  #df_week = data[data.week_start == week_to_plot].copy()
  df_week = data.groupby(['lon','lat'], as_index=False)[outputs+[f'{c}_pred' for c in outputs]].mean()
  
  cmap = 'turbo'
  fig, axes = plt.subplots(2, 3, figsize=(18,7), sharex=True, sharey=True, constrained_layout=True)    
  
  gdf_week = gpd.GeoDataFrame(df_week, geometry=gpd.points_from_xy(df_week.lon, df_week.lat), crs=4326)
  gdf_week['geometry'] = gdf_week.geometry.apply(lambda p: centroid_to_cell(p))
  gdf_week = gdf_week.set_geometry('geometry')
  
  for i, output_var in enumerate(outputs):
    
    gdf_week.plot(ax=axes[0,i%4], column=f"{output_var}_pred", cmap=cmap, vmin=0, vmax=1, missing_kwds={"color": "lightgrey"}, legend=False)
    gdf_week.plot(ax=axes[1,i%4], column=output_var, cmap=cmap, vmin=0, vmax=1, missing_kwds={"color": "lightgrey"}, legend=False)
    world.boundary.plot(ax=axes[0,i%4], color='black')
    world.boundary.plot(ax=axes[1,i%4], color='black')
    
    axes[0,i%4].set_title(f"{output_var} - Predicted")
    axes[0,i%4].set_title(f'{output_var} - Observed')

  for ax in axes.ravel():
    ax.set_xlim(-139, 159)
    ax.set_ylim(-59, 59)
    
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
  sm._A = []
  cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.35, pad=0.02)
  cbar.set_label('Predicted probability', fontsize=15)
  cbar.ax.tick_params(labelsize=15)
  
  plt.savefig(f'./figures/map_multi_{week_tag}.png', dpi=300)
  plt.close(fig)

def plot_multi_spatial_shap(c, df_plot, shap_dfs, outputs, save_path=None):
  world = get_world('world')
  
  fig, axes = plt.subplots(2, 2, figsize=(30,15), layout='constrained')
  vmaxs = np.zeros(4)
  for i, output_var in enumerate(outputs):
    shap_df_temp = shap_dfs[output_var]
    shap_df_temp[['lon','lat']] = df_plot[['lon','lat']]
    shap_df_temp = shap_df_temp.groupby(['lon','lat'], as_index=False)[c].mean()
    shap_df_temp = gpd.GeoDataFrame(shap_df_temp, geometry=gpd.points_from_xy(shap_df_temp.lon, shap_df_temp.lat), crs=4326)
    
    vmax_shap = np.percentile(np.abs(shap_df_temp[c].values), 99)
    vmaxs[i] = vmax_shap
    
    ax = axes[i//2,i%2]
    
    world.boundary.plot(ax=ax, color='black')
    shap_df_temp.plot(ax=ax, column=c, cmap='coolwarm', vmin=-vmax_shap, vmax=vmax_shap, legend=False, marker='s', edgecolor='none', markersize=5)
    if (i//2) == 1: ax.set_xlabel("Longitude", fontsize=15)
    ax.set_ylabel("Latitude", fontsize=15)
    if (i%2) == 1:
      ax.yaxis.tick_right()
      ax.yaxis.set_label_position("right")
    ax.set_xlim(-139, 159)
    ax.set_ylim(-59, 59)
    ax.set_title(output_var, fontsize=20)
    
  sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-max(vmaxs), vmax=max(vmaxs)))
  sm._A = []
  cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", shrink=0.5, pad=0.02)
  cbar.set_label("SHAP value (log expected probability)", fontsize=15)
  cbar.ax.tick_params(labelsize=12)
    
  if save_path is None:
    plt.show()
  else:
    plt.savefig(save_path, dpi=300)


