from typing import Optional, List, Dict, Any
from pathlib import Path
import math
import json
import pandas as pd
from backend.utils.data_loader import data_loader
from backend.schemas.agriculture import (
    SummaryResponse,
    FiltersResponse,
    RecordItem,
    PaginatedRecordsResponse,
    PaginationMetadata,
    TrendPoint,
    TrendsResponse,
    StateSummary,
    StatesResponse,
    DistrictSummary,
    DistrictsResponse,
    CropItem,
    CropsResponse,
    CropDetailResponse,
    CropAvailabilityResponse,
    DatasetMetadataResponse,
)

CROP_WIDE_COLUMN_MAP: Dict[str, Dict[str, str]] = {
    "Rice": {"area": "RICE AREA (1000 ha)", "prod": "RICE PRODUCTION (1000 tons)", "yield": "RICE YIELD (Kg per ha)"},
    "Wheat": {"area": "WHEAT AREA (1000 ha)", "prod": "WHEAT PRODUCTION (1000 tons)", "yield": "WHEAT YIELD (Kg per ha)"},
    "Kharif Sorghum": {"area": "KHARIF SORGHUM AREA (1000 ha)", "prod": "KHARIF SORGHUM PRODUCTION (1000 tons)", "yield": "KHARIF SORGHUM YIELD (Kg per ha)"},
    "Rabi Sorghum": {"area": "RABI SORGHUM AREA (1000 ha)", "prod": "RABI SORGHUM PRODUCTION (1000 tons)", "yield": "RABI SORGHUM YIELD (Kg per ha)"},
    "Sorghum": {"area": "SORGHUM AREA (1000 ha)", "prod": "SORGHUM PRODUCTION (1000 tons)", "yield": "SORGHUM YIELD (Kg per ha)"},
    "Pearl Millet": {"area": "PEARL MILLET AREA (1000 ha)", "prod": "PEARL MILLET PRODUCTION (1000 tons)", "yield": "PEARL MILLET YIELD (Kg per ha)"},
    "Maize": {"area": "MAIZE AREA (1000 ha)", "prod": "MAIZE PRODUCTION (1000 tons)", "yield": "MAIZE YIELD (Kg per ha)"},
    "Finger Millet": {"area": "FINGER MILLET AREA (1000 ha)", "prod": "FINGER MILLET PRODUCTION (1000 tons)", "yield": "FINGER MILLET YIELD (Kg per ha)"},
    "Barley": {"area": "BARLEY AREA (1000 ha)", "prod": "BARLEY PRODUCTION (1000 tons)", "yield": "BARLEY YIELD (Kg per ha)"},
    "Chickpea": {"area": "CHICKPEA AREA (1000 ha)", "prod": "CHICKPEA PRODUCTION (1000 tons)", "yield": "CHICKPEA YIELD (Kg per ha)"},
    "Pigeonpea": {"area": "PIGEONPEA AREA (1000 ha)", "prod": "PIGEONPEA PRODUCTION (1000 tons)", "yield": "PIGEONPEA YIELD (Kg per ha)"},
    "Minor Pulses": {"area": "MINOR PULSES AREA (1000 ha)", "prod": "MINOR PULSES PRODUCTION (1000 tons)", "yield": "MINOR PULSES YIELD (Kg per ha)"},
    "Groundnut": {"area": "GROUNDNUT AREA (1000 ha)", "prod": "GROUNDNUT PRODUCTION (1000 tons)", "yield": "GROUNDNUT YIELD (Kg per ha)"},
    "Sesamum": {"area": "SESAMUM AREA (1000 ha)", "prod": "SESAMUM PRODUCTION (1000 tons)", "yield": "SESAMUM YIELD (Kg per ha)"},
    "Rapeseed and Mustard": {"area": "RAPESEED AND MUSTARD AREA (1000 ha)", "prod": "RAPESEED AND MUSTARD PRODUCTION (1000 tons)", "yield": "RAPESEED AND MUSTARD YIELD (Kg per ha)"},
    "Safflower": {"area": "SAFFLOWER AREA (1000 ha)", "prod": "SAFFLOWER PRODUCTION (1000 tons)", "yield": "SAFFLOWER YIELD (Kg per ha)"},
    "Castor": {"area": "CASTOR AREA (1000 ha)", "prod": "CASTOR PRODUCTION (1000 tons)", "yield": "CASTOR YIELD (Kg per ha)"},
    "Linseed": {"area": "LINSEED AREA (1000 ha)", "prod": "LINSEED PRODUCTION (1000 tons)", "yield": "LINSEED YIELD (Kg per ha)"},
    "Sunflower": {"area": "SUNFLOWER AREA (1000 ha)", "prod": "SUNFLOWER PRODUCTION (1000 tons)", "yield": "SUNFLOWER YIELD (Kg per ha)"},
    "Soyabean": {"area": "SOYABEAN AREA (1000 ha)", "prod": "SOYABEAN PRODUCTION (1000 tons)", "yield": "SOYABEAN YIELD (Kg per ha)"},
    "Oilseeds": {"area": "OILSEEDS AREA (1000 ha)", "prod": "OILSEEDS PRODUCTION (1000 tons)", "yield": "OILSEEDS YIELD (Kg per ha)"},
    "Sugarcane": {"area": "SUGARCANE AREA (1000 ha)", "prod": "SUGARCANE PRODUCTION (1000 tons)", "yield": "SUGARCANE YIELD (Kg per ha)"},
    "Cotton": {"area": "COTTON AREA (1000 ha)", "prod": "COTTON PRODUCTION (1000 tons)", "yield": "COTTON YIELD (Kg per ha)"},
    "Fruits": {"area": "FRUITS AREA (1000 ha)", "prod": None, "yield": None},
    "Vegetables": {"area": "VEGETABLES AREA (1000 ha)", "prod": None, "yield": None},
    "Fruits and Vegetables": {"area": "FRUITS AND VEGETABLES AREA (1000 ha)", "prod": None, "yield": None},
    "Potatoes": {"area": "POTATOES AREA (1000 ha)", "prod": None, "yield": None},
    "Onion": {"area": "ONION AREA (1000 ha)", "prod": None, "yield": None},
    "Fodder": {"area": "FODDER AREA (1000 ha)", "prod": None, "yield": None}
}

class AgricultureService:
    _panel_df: Optional[pd.DataFrame] = None

    def get_df(self) -> pd.DataFrame:
        return data_loader.dataframe

    def get_panel_df(self) -> pd.DataFrame:
        """Loads and caches the unified multi-crop agricultural panel dataset."""
        if self._panel_df is not None:
            return self._panel_df
        panel_path = Path("datasets/processed/agricultural_panel.csv")
        if panel_path.exists():
            self._panel_df = pd.read_csv(panel_path)
            return self._panel_df
        return pd.DataFrame()

    def _resolve_crop_columns(self, crop: Optional[str] = None) -> tuple[str, Optional[str], Optional[str]]:
        std_crop = "Rice"
        if crop:
            for k in CROP_WIDE_COLUMN_MAP:
                if k.lower() == crop.strip().lower():
                    std_crop = k
                    break
        cmap = CROP_WIDE_COLUMN_MAP.get(std_crop, CROP_WIDE_COLUMN_MAP["Rice"])
        return cmap["area"], cmap["prod"], cmap["yield"]

    def get_crops(self) -> CropsResponse:
        """Returns the complete list of verified multi-crop agricultural commodities."""
        df = self.get_df()
        crops_list: List[CropItem] = []
        
        for crop_name, cmap in sorted(CROP_WIDE_COLUMN_MAP.items()):
            a_col = cmap["area"]
            p_col = cmap["prod"]
            y_col = cmap["yield"]
            
            has_area = a_col is not None and a_col in df.columns
            has_prod = p_col is not None and p_col in df.columns
            has_yield = y_col is not None and y_col in df.columns
            
            # Count records with non-null positive observation
            recs = 0
            states_cnt = 0
            dists_cnt = 0
            if has_yield and y_col:
                valid = df[df[y_col].notnull() & (df[y_col] >= 0)]
                recs = len(valid)
                states_cnt = int(valid['State Name'].nunique()) if recs > 0 else 0
                dists_cnt = int(valid['Dist Name'].nunique()) if recs > 0 else 0
            elif has_area and a_col:
                valid = df[df[a_col].notnull() & (df[a_col] >= 0)]
                recs = len(valid)
                states_cnt = int(valid['State Name'].nunique()) if recs > 0 else 0
                dists_cnt = int(valid['Dist Name'].nunique()) if recs > 0 else 0

            is_rice = (crop_name.lower() == "rice")
            crops_list.append(
                CropItem(
                    crop=crop_name,
                    records=recs if recs > 0 else len(df),
                    first_year=int(df['Year'].min()),
                    last_year=int(df['Year'].max()),
                    states=states_cnt if states_cnt > 0 else int(df['State Name'].nunique()),
                    districts=dists_cnt if dists_cnt > 0 else int(df['Dist Name'].nunique()),
                    has_area=has_area,
                    has_production=has_prod,
                    has_yield=has_yield,
                    forecasting_supported=is_rice,
                    scenario_supported=is_rice,
                    xai_supported=is_rice
                )
            )
        return CropsResponse(total_crops=len(crops_list), crops=crops_list)

    def get_crop_detail(self, crop: str) -> CropDetailResponse:
        """Returns comprehensive historical metadata and distribution for a specific crop."""
        df = self.get_df()
        std_crop = "Rice"
        for k in CROP_WIDE_COLUMN_MAP:
            if k.lower() == crop.strip().lower():
                std_crop = k
                break
        
        a_col, p_col, y_col = self._resolve_crop_columns(std_crop)
        valid = df
        if y_col and y_col in df.columns:
            valid = df[df[y_col].notnull() & (df[y_col] >= 0)]
        elif a_col and a_col in df.columns:
            valid = df[df[a_col].notnull() & (df[a_col] >= 0)]

        total_prod = float(round(valid[p_col].sum() * 1000.0, 2)) if p_col and p_col in valid.columns else 0.0
        avg_yield = float(round(valid[y_col].mean(), 2)) if y_col and y_col in valid.columns and len(valid) > 0 else 0.0
        is_rice = (std_crop.lower() == "rice")

        return CropDetailResponse(
            crop=std_crop,
            records=len(valid),
            first_year=int(valid['Year'].min()) if len(valid) > 0 else int(df['Year'].min()),
            last_year=int(valid['Year'].max()) if len(valid) > 0 else int(df['Year'].max()),
            states=sorted(valid['State Name'].dropna().unique().tolist()),
            districts_count=int(valid['Dist Name'].nunique()),
            total_production_tonnes=total_prod,
            average_yield_kg_ha=avg_yield,
            forecasting_model_status="REGISTERED_AND_VALIDATED" if is_rice else "HISTORICAL_ANALYTICS_ONLY",
            available_years=sorted(valid['Year'].unique().tolist())
        )

    def get_availability(
        self,
        crop: Optional[str] = "Rice",
        state: Optional[str] = None,
        district: Optional[str] = None,
        year: Optional[int] = None,
        season: Optional[str] = None
    ) -> CropAvailabilityResponse:
        """Evaluates whether the requested combination of crop and filters has verified historical records."""
        df = self.get_df()
        std_crop = "Rice"
        if crop:
            for k in CROP_WIDE_COLUMN_MAP:
                if k.lower() == crop.strip().lower():
                    std_crop = k
                    break
        
        a_col, p_col, y_col = self._resolve_crop_columns(std_crop)
        filtered = df
        
        # Crop filter
        if y_col and y_col in df.columns:
            filtered = filtered[filtered[y_col].notnull() & (filtered[y_col] >= 0)]
        elif a_col and a_col in df.columns:
            filtered = filtered[filtered[a_col].notnull() & (filtered[a_col] >= 0)]

        state_avail = True
        dist_avail = True
        year_avail = True

        if state and state.lower() != 'all':
            s_match = filtered[filtered['State Name'].str.lower() == state.lower()]
            state_avail = len(s_match) > 0
            filtered = s_match

        if district and district.lower() != 'all':
            d_match = filtered[filtered['Dist Name'].str.lower() == district.lower()]
            dist_avail = len(d_match) > 0
            filtered = d_match

        if year:
            y_match = filtered[filtered['Year'] == year]
            year_avail = len(y_match) > 0
            filtered = y_match

        is_rice = (std_crop.lower() == "rice")
        models = ["RandomForest_PostHarvest", "RandomForest_PreSeason", "Advanced_Exogenous_RF", "IsolationForest_Anomaly"] if is_rice else []

        return CropAvailabilityResponse(
            crop=std_crop,
            available=len(filtered) > 0,
            state_available=state_avail,
            district_available=dist_avail,
            year_available=year_avail,
            matching_records=len(filtered),
            supported_models=models
        )

    def get_metadata(self) -> DatasetMetadataResponse:
        """Returns the canonical dataset metadata and governance manifest."""
        manifest_path = Path("datasets/metadata/dataset_manifest.json")
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                data = json.load(f)
                return DatasetMetadataResponse(
                    dataset_version=data.get("dataset_version", "AGRI_PANEL_1.0"),
                    sources=data.get("sources", ["ICRISAT_DLD_1966_2017"]),
                    record_count=data.get("records", 71601),
                    crop_count=data.get("crop_count", 29),
                    state_count=data.get("state_count", 20),
                    district_count=data.get("districts_count", 311),
                    year_range=f"{data.get('years', [2010, 2017])[0]}-{data.get('years', [2010, 2017])[1]}",
                    available_crops=data.get("crops", list(CROP_WIDE_COLUMN_MAP.keys())),
                    available_states=data.get("states", []),
                    quality_status=data.get("quality_status", "PASS")
                )
        df = self.get_df()
        return DatasetMetadataResponse(
            dataset_version="AGRI_PANEL_1.0",
            sources=["ICRISAT_DLD_1966_2017"],
            record_count=len(df),
            crop_count=len(CROP_WIDE_COLUMN_MAP),
            state_count=int(df['State Name'].nunique()),
            district_count=int(df['Dist Name'].nunique()),
            year_range=f"{df['Year'].min()}-{df['Year'].max()}",
            available_crops=list(CROP_WIDE_COLUMN_MAP.keys()),
            available_states=sorted(df['State Name'].unique().tolist()),
            quality_status="PASS"
        )

    def get_summary(self, crop: Optional[str] = "Rice") -> SummaryResponse:
        df = self.get_df()
        a_col, p_col, y_col = self._resolve_crop_columns(crop)

        # Calculate exact statistics
        total_records = len(df)
        total_states = int(df['State Name'].nunique())
        total_districts = int(df['Dist Name'].nunique())
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())

        average_yield = float(round(df[y_col].mean(), 2)) if (y_col and y_col in df.columns) else 0.0
        median_yield = float(round(df[y_col].median(), 2)) if (y_col and y_col in df.columns) else 0.0
        average_area = float(round(df[a_col].mean(), 2)) if (a_col and a_col in df.columns) else 0.0
        average_production = float(round(df[p_col].mean(), 2)) if (p_col and p_col in df.columns) else 0.0
        total_area = float(round(df[a_col].sum(), 2)) if (a_col and a_col in df.columns) else 0.0
        total_production = float(round(df[p_col].sum(), 2)) if (p_col and p_col in df.columns) else 0.0
        zero_yield_records = int((df[y_col] == 0).sum()) if (y_col and y_col in df.columns) else 0
        missing_values = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        columns_count = int(len(df.columns))

        return SummaryResponse(
            total_records=total_records,
            total_states=total_states,
            total_districts=total_districts,
            min_year=min_year,
            max_year=max_year,
            average_yield=average_yield,
            median_yield=median_yield,
            average_area=average_area,
            average_production=average_production,
            total_area=total_area,
            total_production=total_production,
            zero_yield_records=zero_yield_records,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            columns_count=columns_count,
        )

    def get_filters(self, crop: Optional[str] = None) -> FiltersResponse:
        df = self.get_df()
        years = sorted([int(y) for y in df['Year'].unique()], reverse=True)
        states = sorted([str(s) for s in df['State Name'].unique()])
        districts = sorted([str(d) for d in df['Dist Name'].unique()])
        crops = sorted(list(CROP_WIDE_COLUMN_MAP.keys()))

        return FiltersResponse(
            years=years,
            states=states,
            districts=districts,
            crops=crops,
        )

    def get_records(
        self,
        page: int = 1,
        page_size: int = 20,
        year: Optional[int] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        search: Optional[str] = None,
        crop: Optional[str] = "Rice",
    ) -> PaginatedRecordsResponse:
        df = self.get_df()
        a_col, p_col, y_col = self._resolve_crop_columns(crop)

        # Apply filters
        filtered = df
        if year is not None:
            filtered = filtered[filtered['Year'] == year]
        if state is not None and state != 'all':
            filtered = filtered[filtered['State Name'].str.lower() == state.lower()]
        if district is not None and district != 'all':
            filtered = filtered[filtered['Dist Name'].str.lower() == district.lower()]
        if search:
            s = search.lower()
            filtered = filtered[
                filtered['State Name'].str.lower().str.contains(s) |
                filtered['Dist Name'].str.lower().str.contains(s)
            ]

        total = len(filtered)
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        total_pages = math.ceil(total / page_size) if total > 0 else 1

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        sliced = filtered.iloc[start_idx:end_idx]

        records: List[RecordItem] = []
        for idx, row in sliced.iterrows():
            rec_id = f"rec-{row.get('Dist Code', idx)}-{row.get('Year', 0)}-{idx}"
            area_v = float(round(row[a_col], 2)) if (a_col and a_col in row and pd.notnull(row[a_col])) else 0.0
            prod_v = float(round(row[p_col], 2)) if (p_col and p_col in row and pd.notnull(row[p_col])) else 0.0
            yield_v = float(round(row[y_col], 2)) if (y_col and y_col in row and pd.notnull(row[y_col])) else 0.0

            records.append(
                RecordItem(
                    id=rec_id,
                    state=str(row['State Name']),
                    district=str(row['Dist Name']),
                    year=int(row['Year']),
                    area=area_v,
                    production=prod_v,
                    yield_val=yield_v,
                )
            )

        return PaginatedRecordsResponse(
            data=records,
            pagination=PaginationMetadata(
                page=page,
                page_size=page_size,
                total=total,
                total_pages=total_pages,
            ),
        )

    def get_trends(
        self,
        state: Optional[str] = None,
        district: Optional[str] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        crop: Optional[str] = "Rice",
    ) -> TrendsResponse:
        df = self.get_df()
        a_col, p_col, y_col = self._resolve_crop_columns(crop)

        filtered = df
        if state is not None and state != 'all':
            filtered = filtered[filtered['State Name'].str.lower() == state.lower()]
        if district is not None and district != 'all':
            filtered = filtered[filtered['Dist Name'].str.lower() == district.lower()]
        if year_start is not None:
            filtered = filtered[filtered['Year'] >= year_start]
        if year_end is not None:
            filtered = filtered[filtered['Year'] <= year_end]

        if filtered.empty:
            return TrendsResponse(data=[])

        # Aggregate safely
        agg_dict = {}
        if y_col and y_col in filtered.columns:
            agg_dict['average_yield'] = (y_col, 'mean')
            agg_dict['record_count'] = (y_col, 'count')
        if a_col and a_col in filtered.columns:
            agg_dict['average_area'] = (a_col, 'mean')
            agg_dict['total_area'] = (a_col, 'sum')
        if p_col and p_col in filtered.columns:
            agg_dict['average_production'] = (p_col, 'mean')
            agg_dict['total_production'] = (p_col, 'sum')

        grouped = filtered.groupby('Year').agg(**agg_dict).reset_index().sort_values('Year')

        trends = []
        for _, row in grouped.iterrows():
            trends.append(
                TrendPoint(
                    year=int(row['Year']),
                    average_yield=float(round(row.get('average_yield', 0.0), 2)),
                    average_area=float(round(row.get('average_area', 0.0), 2)),
                    average_production=float(round(row.get('average_production', 0.0), 2)),
                    total_production=float(round(row.get('total_production', 0.0), 2)),
                    total_area=float(round(row.get('total_area', 0.0), 2)),
                    record_count=int(row.get('record_count', len(filtered))),
                )
            )

        return TrendsResponse(data=trends)

    def get_states(self, crop: Optional[str] = "Rice") -> StatesResponse:
        df = self.get_df()
        a_col, p_col, y_col = self._resolve_crop_columns(crop)

        agg_dict = {
            'district_count': ('Dist Name', 'nunique')
        }
        if y_col and y_col in df.columns:
            agg_dict['record_count'] = (y_col, 'count')
            agg_dict['average_yield'] = (y_col, 'mean')
            agg_dict['median_yield'] = (y_col, 'median')
        if a_col and a_col in df.columns:
            agg_dict['total_area'] = (a_col, 'sum')
        if p_col and p_col in df.columns:
            agg_dict['total_production'] = (p_col, 'sum')

        grouped = df.groupby('State Name').agg(**agg_dict).reset_index()
        sort_col = 'average_yield' if 'average_yield' in grouped.columns else 'State Name'
        grouped = grouped.sort_values(sort_col, ascending=False)

        states: List[StateSummary] = []
        for rank, (_, row) in enumerate(grouped.iterrows(), start=1):
            states.append(
                StateSummary(
                    state=str(row['State Name']),
                    rank=rank,
                    record_count=int(row.get('record_count', 0)),
                    district_count=int(row['district_count']),
                    average_yield=float(round(row.get('average_yield', 0.0), 2)),
                    median_yield=float(round(row.get('median_yield', 0.0), 2)),
                    total_area=float(round(row.get('total_area', 0.0), 2)),
                    total_production=float(round(row.get('total_production', 0.0), 2)),
                )
            )

        return StatesResponse(data=states)

    def get_districts(
        self,
        state: Optional[str] = None,
        year: Optional[int] = None,
        crop: Optional[str] = "Rice",
    ) -> DistrictsResponse:
        df = self.get_df()
        a_col, p_col, y_col = self._resolve_crop_columns(crop)

        filtered = df
        if state is not None and state != 'all':
            filtered = filtered[filtered['State Name'].str.lower() == state.lower()]
        if year is not None:
            filtered = filtered[filtered['Year'] == year]

        districts: List[DistrictSummary] = []
        for _, row in filtered.iterrows():
            area_v = float(round(row[a_col], 2)) if (a_col and a_col in row and pd.notnull(row[a_col])) else 0.0
            prod_v = float(round(row[p_col], 2)) if (p_col and p_col in row and pd.notnull(row[p_col])) else 0.0
            yield_v = float(round(row[y_col], 2)) if (y_col and y_col in row and pd.notnull(row[y_col])) else 0.0

            districts.append(
                DistrictSummary(
                    district=str(row['Dist Name']),
                    state=str(row['State Name']),
                    year=int(row['Year']),
                    area=area_v,
                    production=prod_v,
                    yield_val=yield_v,
                )
            )

        return DistrictsResponse(data=districts)


agriculture_service = AgricultureService()
