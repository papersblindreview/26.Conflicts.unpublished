# Data

No data are stored in this repository. This file says where to get each input and where to save it (paths are relative to the repository root, and are the paths the scripts read).
Check each provider's license and terms before redistributing anything; ACLED in particular does not allow redistribution of its raw data.

| Group | Inputs | Obtained by |
|---|---|---|
| [Conflict events](#conflict-events-acled) | ACLED | manual download (free account) |
| [Weekly climate](#weekly-climate) | CHIRPS, CHIRTS-ERA5, CAMS PM2.5 | script (`preprocessing.py --stage download`) |
| [Drought](#drought) | Global Drought Observatory | manual download |
| [Geography](#geography-and-boundaries) | countries, rivers, lakes, biomes, admin boundaries, population | manual download |
| [Socioeconomic and governance](#socioeconomic-governance-and-sustainability) | GDP, electricity, food insecurity, malaria, EPI, V-Dem, GHG / material / ecological footprints | manual download |

Everything under `data/` is ignored by git.

---

## Conflict events (ACLED)

- **Where:** Armed Conflict Location & Event Data Project, <https://acleddata.com> (data export tool; requires a free account and acceptance of the terms of use).
- **What to download:** all event types, all countries, **2018-01-01 to 2024-12-31**, exported as CSV with **all columns**. Several files are fine (for example one per year or region); every CSV in the folder is concatenated and exact duplicates are dropped.
- **Save as:** `data/acled/*.csv`
- **Columns the code needs:** `event_date`, `event_type`, `sub_event_type`, `disorder_type`, `latitude`, `longitude`. The loader also drops a fixed list of other ACLED columns by name (`source`, `source_scale`, `notes`, `actor1`, `assoc_actor_1`, `inter1`, `actor2`, `assoc_actor_2`, `inter2`, `iso`, `region`, `tags`, `timestamp`, `country`, `year`, `admin1`, `admin2`, `admin3`, `location`, `geo_precision`, `time_precision`, `event_id_cnty`, `interaction`, `civilian_targeting`, `fatalities`, `population_best`), so an export that omits any of them will fail.

## Weekly climate

All three are downloaded and averaged to weeks by one command:

```bash
python scripts/preprocessing.py --stage download --var all     # or one of: ppt, tmax, tmin, hi, wbgt, pm25
```

Daily files are downloaded, averaged over each 7-day block starting 1 January, written as compressed NetCDF, and the daily files are deleted. Expect this to take a long time and a lot of bandwidth.

| Variable | Source | Saved as |
|---|---|---|
| Precipitation | CHIRPS v3.0 daily final, <https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/> (UCSB Climate Hazards Center) | `data/CHIRPS/ppt_weekly_<YYYY.MM.DD>.nc` |
| Minimum and maximum temperature, heat index, wet-bulb globe temperature | CHIRTS-ERA5 daily, <https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/> | `data/CHIRTS/<var>/<var>_weekly_<YYYY.MM.DD>.nc` for `tmax`, `tmin`, `hi`, `wbgt` |
| PM2.5 | CAMS global reanalysis (EAC4), 12:00 UTC at model level 60, from the Copernicus Atmosphere Data Store, <https://ads.atmosphere.copernicus.eu> | `data/CAMS/pm25/pm25_weekly_<YYYY.MM.DD>.nc` |

**CAMS setup (one time):** create a free account at the Atmosphere Data Store, open the dataset page "CAMS global reanalysis (EAC4)" and accept its license, then put your personal access token in `~/.cdsapirc`:

```
url: https://ads.atmosphere.copernicus.eu/api
key: <your-personal-access-token>
```

(The EAC4 data are served by the Atmosphere Data Store, not the Climate Data Store, so the `url` line matters.) The CAMS step also needs the `cfgrib` package and the ecCodes library.

## Drought

- **Where:** Global Drought Observatory, Copernicus Emergency Management Service, <https://drought.emergency.copernicus.eu>. Download the global drought-indicator rasters (GeoTIFF) for 2018-2024.
- **Save as:** `data/drought/<year>/<files>.tif`, one folder per year.
- **File-name rule:** the date is read from the fourth underscore-separated part of the file name, as `YYYYMMDD` (for example `abc_x_glb_20180101_t.tif`), so keep the provider's original names. Values are averaged to the 0.5-degree grid and to weeks.

## Geography and boundaries

| Input | Where | Save as |
|---|---|---|
| Country boundaries | Natural Earth, 1:10m Admin 0 Countries, <https://www.naturalearthdata.com/downloads/10m-cultural-vectors/> | `data/world_map/ne_10m_admin_0_countries.shp` (with its `.dbf`, `.shx`, `.prj`) |
| Rivers | Natural Earth, 1:10m Rivers + Lake Centerlines, <https://www.naturalearthdata.com/downloads/10m-physical-vectors/> | `data/rivers/ne_10m_rivers_lake_centerlines.shp` |
| Lakes | Natural Earth, 1:10m Lakes + Reservoirs, same page | `data/lakes/ne_10m_lakes.shp` |
| Biomes | RESOLVE Ecoregions 2017 (Dinerstein et al. 2017, *BioScience*), <https://ecoregions.appspot.com>; the shapefile's `BIOME_NAME` field is used | `data/biomes/Ecoregions2017.shp` |
| Country boundaries for joining national indicators | World Bank Official Boundaries, <https://datacatalog.worldbank.org/search/dataset/0038272/world-bank-official-boundaries> (CC BY 4.0); download the shapefile resource and keep the admin-2 layer, which the code dissolves to countries by `ISO_A3` | `data/world_bank/WB_GAD_ADM2.shp` |
| Population | GHSL, GHS-POP R2023A, epoch 2010, WGS84, 30 arc-second, <https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php> (file index: <https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2010_GLOBE_R2023A_4326_30ss/V1-0/>); unzip the GeoTIFF | `data/population/GHS_POP_E2010_GLOBE_R2023A_4326_30ss_V1_0.tif` |

## Socioeconomic, governance, and sustainability

All files below go in `data/socioeconomic/`.

**GDP** → `GDP.csv`
World Bank World Development Indicators, GDP, <https://data.worldbank.org/indicator/NY.GDP.MKTP.CD> (download as CSV). The code reads the file with default settings, so the **first line must be the column header**: delete the four metadata lines at the top of the World Bank download. Required columns: `Country Name`, `Country Code`, `Indicator Name`, `Indicator Code`, then one column per year (`2018` ... `2024`). Country GDP is later shared across grid cells in proportion to population.

**Electricity access** → `electricity.csv`
SDG Indicator 7.1.1, Access to Electricity, 2023 Release, NASA SEDAC (CIESIN), DOI [10.7927/gxnr-sx57](https://doi.org/10.7927/gxnr-sx57), available at national and subnational level 2. The code uses the columns `ISO3`, `Pop_A2` (population) and `SDG711pct` (percent with access) and aggregates them to a population-weighted national share; all other columns after the fifth are treated as numeric and negative values are set to missing.

**Food insecurity** → `hdx_hapi_food_security_global.csv` and `global_admin_boundaries_matched_latest.gdb`
IPC acute food insecurity phases, distributed through the Humanitarian Data Exchange HAPI (<https://docs.humdata.org/build/hdx-apis/hapi/api-reference/food-security-nutrition-and-poverty>; upstream source: <https://data.humdata.org/dataset/global-acute-food-insecurity-country-data>). The CSV needs the columns `location_code`, `admin1_code`, `admin_level`, `ipc_type`, `ipc_phase`, `population_in_phase`, `reference_period_start` and `reference_period_end`; the code keeps `ipc_type == current`, admin level 1, and phases 1-5. The geodatabase must contain a layer named `admin1` with the columns `iso3` and `adm1_pcode` (OCHA administrative boundary P-codes matching the `admin1_code` values in the CSV). Population in each phase is shared across grid cells in proportion to population.

**Malaria** → `malaria/`
Malaria Atlas Project yearly prevalence rasters (<https://malariaatlas.org>; the R package `malariaAtlas` downloads them with `getRaster()`, for example the *Plasmodium falciparum* parasite rate surface), one GeoTIFF per year, 2018-2024. The year is read from the last underscore-separated part of each file name (it must start with the four-digit year, as in `..._2018.tif`), and `-9999` is treated as no data.

**Environmental Performance Index** → `EPI/`
Yale Environmental Performance Index 2024, <https://epi.yale.edu/downloads>. From the 2024 downloads, put the **Weights (CSV)** file in as `EPI/epi2024weights.csv`, and unzip the **Indicator scores (ZIP)** into the same folder. The code reads every file whose name contains `ind_na`, keeps years 2018 and later, and recomputes the index as the weighted mean of the indicator scores using `epi2024weights.csv`.

**Liberal democracy index** → `vdem_libdem_index.csv`
Varieties of Democracy, <https://v-dem.net/data/the-v-dem-dataset/>. Download the Country-Year dataset (use a release that covers 2024), keep the three columns `country_text_id`, `year` and `v2x_libdem`, rename `country_text_id` to `ISO_A3`, and save as CSV.

**Greenhouse gas, material footprint, ecological footprint (per person)** → three CSVs
National annual tables for 1970-2024, each with the columns `Country`, `ISO3`, `YEAR` and one value column:

| File | Value column | Quantity | Source |
|---|---|---|---|
| `GHG_per_capita_t_1970_2024.csv` | `GHG_pc_t` | greenhouse gas emissions, tons per person | national emissions per capita, as cited in the manuscript |
| `MF_per_capita_t_1970_2024.csv` | `MF_pc_t` | material footprint, tons per person | UNEP International Resource Panel, Global Material Flows Database, <https://www.resourcepanel.org/global-material-flows-database> |
| `ECOLOGICAL_FOOTPRINT_per_capita_1970_2024.csv` | `ECO_pc_gha` | ecological footprint, global hectares per person | Global Footprint Network, National Footprint and Biocapacity Accounts, <https://data.footprintnetwork.org> |

The three tables are merged on `Country`, `ISO3` and `YEAR`, so the country names must be spelled identically across files. These values are national and constant within a country-year.

---

## Files produced by the pipeline

- `data/partitions/data_final_world_XX.csv` (XX = 01..50): the grid-cell-week dataset, one file per block of dates (`scripts/preprocessing.py`).
- `results/plot_data_01_{Conflict,Unrest}.csv`: the fixed explanation sample of up to 300,000 rows (`scripts/model_xgb.py`).
- `results/shap_XGB_01_{Conflict,Unrest}.csv`: SHAP values of the fitted model on that sample (`scripts/model_xgb.py`).
- `results/boot/shap_<Outcome>_<rep>.csv`: one file per bootstrap replicate (`scripts/bootstrap_shap.py`).
