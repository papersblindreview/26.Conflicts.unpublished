# Data

Raw and intermediate data are not stored in this repository. Place the inputs below under `data/` using these paths (they are the paths the scripts read).
Check each provider's terms before redistributing; ACLED in particular requires registration.

## Event data

| Path | Source |
|---|---|
| `data/acled/*.csv` | Armed Conflict Location & Event Data Project (ACLED), event-level exports covering 2018-2024 |

## Climate (weekly means, produced by `scripts/preprocessing.py --stage download`)

| Path | Source | Download option |
|---|---|---|
| `data/CHIRPS/ppt_weekly_*.nc` | CHIRPS v3.0 daily precipitation (UCSB Climate Hazards Center) | `--var ppt` |
| `data/CHIRTS/{tmax,tmin,hi,wbgt}/*_weekly_*.nc` | CHIRTS-ERA5 daily temperature, heat index, wet-bulb globe temperature | `--var tmax`, `tmin`, `hi`, `wbgt` |
| `data/CAMS/pm25/pm25_weekly_*.nc` | CAMS global reanalysis (EAC4) PM2.5 via the Copernicus Climate Data Store | `--var pm25` |
| `data/drought/<year>/*` | Global Drought Observatory (Copernicus) drought-indicator rasters | manual |

## Boundaries and geography

| Path | Source |
|---|---|
| `data/world_map/ne_10m_admin_0_countries.shp` | Natural Earth, admin-0 countries |
| `data/rivers/ne_10m_rivers_lake_centerlines.shp`, `data/lakes/ne_10m_lakes.shp` | Natural Earth |
| `data/biomes/Ecoregions2017.shp` | RESOLVE Ecoregions 2017 (biomes) |
| `data/world_bank/WB_GAD_ADM2.shp` | World Bank Global Administrative Divisions (dissolved to countries) |
| `data/population/GHS_POP_E2010_GLOBE_R2023A_4326_30ss_V1_0.tif` | GHSL GHS-POP, 2010, 30 arc-second |

## Socioeconomic, governance and sustainability (`data/socioeconomic/`)

| Path | Contents |
|---|---|
| `GDP.csv` | World Bank GDP (wide format, one column per year, `Country Code`) |
| `electricity.csv` | Gridded population and SDG 7.1.1 electricity access (`ISO3`, `Pop_A2`, `SDG711pct`) |
| `hdx_hapi_food_security_global.csv`, `global_admin_boundaries_matched_latest.gdb` | IPC food-insecurity phases (HDX HAPI) and matching admin-1 boundaries |
| `malaria/*` | Malaria Atlas Project yearly rasters |
| `EPI/` | Yale Environmental Performance Index 2024 indicator files and `epi2024weights.csv` |
| `vdem_libdem_index.csv` | V-Dem liberal democracy index (`ISO_A3`, `year`, `v2x_libdem`) |
| `GHG_per_capita_t_1970_2024.csv`, `MF_per_capita_t_1970_2024.csv`, `ECOLOGICAL_FOOTPRINT_per_capita_1970_2024.csv` | National per-capita greenhouse gas emissions, material footprint, and ecological footprint (columns `Country`, `ISO3`, `YEAR`, and the value column; sources as cited in the manuscript) |

## Intermediate files written by the pipeline

- `data/partitions/data_final_world_XX.csv` (XX = 01..50): the grid-cell-week dataset, one file per block of dates.
- `results/plot_data_01_{Conflict,Unrest}.csv`: the fixed explanation sample (about 300,000 rows) written by `scripts/model_xgb.py`.
- `results/boot/shap_<Outcome>_<rep>.csv`: one file per bootstrap replicate.
