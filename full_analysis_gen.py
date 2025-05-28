#!/usr/bin/env python3
"""
Advanced CFIA Food Recall Analysis Script - V3
==============================================

This script performs comprehensive analysis of standardized CFIA recall data.
It leverages the known data structure and batch-nature of recalls to generate
research-focused insights, ML-ready features emphasizing generalization,
and data for network analysis.

Requirements:
    pip install pandas numpy openpyxl scikit-learn

Usage:
    python cfia_advanced_analysis_v3.py <excel_file> [output_directory]

Outputs:
    - (Multiple CSVs as before)
    - cfia_enhanced_dataset_ml.csv (Now includes incident-level features)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import logging
import re
from collections import Counter
from itertools import combinations
import argparse # Import argparse

# --- Configuration ---

# Standardized column names based on user input.
# Keeping a list allows minor variations (e.g., if case changes).
COLUMN_MAPPING = {
    'RECALL_DATE': ['RECALL DATE', 'RECALL_DATE'],
    'RECALL_NUMBER': ['RECALL NUMBER', 'RECALL_NUMBER'],
    'RECALL_CLASS': ['RECALL CLASS', 'RECALL_CLASS'],
    'AREA_OF_CONCERN': ['AREA OF CONCERN', 'AREA_OF_CONCERN'],
    'PRIMARY_RECALL': ['PRIMARY RECALL?', 'PRIMARY_RECALL'],
    'DEPTH': ['DEPTH'],
    'BRAND_NAME': ['BRAND NAME', 'BRAND_NAME'],
    'COMMON_NAME': ['COMMON NAME', 'COMMON_NAME'],
}

CLASS_NORMALIZATION = {
    r'\bclass\s*i\b': 'Class I',
    r'\bclass\s*1\b': 'Class I',
    r'\bclass\s*ii\b': 'Class II',
    r'\bclass\s*2\b': 'Class II',
    r'\bclass\s*iii\b': 'Class III',
    r'\bclass\s*3\b': 'Class III',
}

# Setup Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

class CFIAAdvancedAnalyzerV3:
    """ Analyzes CFIA Food Recall Data - V3 """
    def __init__(self, excel_file):
        logging.info("Starting Advanced CFIA Recall Analysis (V3)...")
        logging.info("=" * 60)
        self.excel_file = excel_file
        self.df = None
        self.incident_df = None # Ensure incident_df is initialized
        self.load_and_prepare_data()

    def _find_column(self, df_columns, standard_name):
        """Finds the actual column name from a list of possibilities."""
        potential_names = COLUMN_MAPPING.get(standard_name, [standard_name])
        for name_variant in potential_names:
            for col in df_columns:
                # Be more flexible: strip, upper, replace space/underscore
                normalized_col = col.strip().upper().replace(' ', '_').replace('?', '')
                normalized_variant = name_variant.strip().upper().replace(' ', '_').replace('?', '')
                if normalized_col == normalized_variant:
                    return col
        return None

    def _normalize_column_names(self):
        """Standardizes column names based on COLUMN_MAPPING."""
        original_columns = list(self.df.columns)
        new_names = {}
        processed_originals = set()

        for std_name in COLUMN_MAPPING.keys():
            actual_col = self._find_column(original_columns, std_name)
            if actual_col and actual_col not in processed_originals:
                new_names[actual_col] = std_name
                processed_originals.add(actual_col)

        # Rename found columns
        self.df.rename(columns=new_names, inplace=True)

        # Standardize *all* columns (including those not in mapping)
        self.df.columns = [
            col.strip().upper().replace(' ', '_').replace('?', '')
            for col in self.df.columns
        ]

        logging.info(f"Standardized columns to: {list(self.df.columns)}")
        essential = ['RECALL_DATE', 'BRAND_NAME', 'COMMON_NAME', 'RECALL_CLASS', 'AREA_OF_CONCERN', 'RECALL_NUMBER']
        for col in essential:
            if col not in self.df.columns:
                 logging.critical(f"Essential column '{col}' not found. Check input or MAPPING. Exiting.")
                 sys.exit(1)


    def _normalize_class(self, value):
        """Normalizes Recall Class values."""
        value_str = str(value).strip().lower()
        for pattern, normalized in CLASS_NORMALIZATION.items():
            if re.search(pattern, value_str):
                return normalized
        return 'Unknown' if pd.isna(value) or value_str == 'nan' else str(value).strip()

    def load_and_prepare_data(self):
        """Loads data, cleans it, and adds basic features."""
        logging.info(f"Loading data from {self.excel_file}...")
        try:
            # Handle potential datetime format issues during loading
            self.df = pd.read_excel(self.excel_file)
            logging.info(f"Loaded {len(self.df):,} records.")
        except FileNotFoundError:
            logging.error(f"Error: File not found at {self.excel_file}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading Excel file: {e}")
            sys.exit(1)

        self.df = self.df.dropna(how='all')
        self._normalize_column_names()

        # Date Handling - address potential time component
        self.df['RECALL_DATE'] = pd.to_datetime(self.df['RECALL_DATE'], errors='coerce')
        self.df = self.df.dropna(subset=['RECALL_DATE'])
        self.df = self.df.sort_values('RECALL_DATE').reset_index(drop=True)

        # Basic Time Features
        self.df['YEAR'] = self.df['RECALL_DATE'].dt.year
        self.df['MONTH'] = self.df['RECALL_DATE'].dt.month
        self.df['WEEK'] = self.df['RECALL_DATE'].dt.isocalendar().week.astype(int)
        self.df['DAY_OF_WEEK'] = self.df['RECALL_DATE'].dt.day_name()
        self.df['QUARTER'] = self.df['RECALL_DATE'].dt.quarter
        self.df['SEASON'] = self.df['MONTH'].map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Fall',10:'Fall',11:'Fall'})

        # Text/Categorical Cleaning
        for col in ['AREA_OF_CONCERN', 'BRAND_NAME', 'COMMON_NAME', 'DEPTH']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.lower().replace('nan', 'unknown')

        self.df['RECALL_CLASS'] = self.df['RECALL_CLASS'].apply(self._normalize_class)
        self.df['RECALL_NUMBER'] = self.df['RECALL_NUMBER'].astype(str).str.strip()
        logging.info("Data loading and preparation complete.")

    # --- Analysis Functions (largely unchanged, ensure _get_mode & _get_severity_counts exist) ---
    def _get_mode(self, series):
        """Safely gets the mode (most frequent value) or 'N/A'."""
        modes = series.mode()
        return modes.iloc[0] if not modes.empty else 'N/A'

    def _get_severity_counts(self, group):
        """Calculates counts and percentages for each recall class."""
        total = len(group)
        severity_counts = group['RECALL_CLASS'].value_counts()
        result = {
            'ClassI': severity_counts.get('Class I', 0),
            'ClassII': severity_counts.get('Class II', 0),
            'ClassIII': severity_counts.get('Class III', 0),
            'UnknownClass': severity_counts.get('Unknown', 0),
        }
        result.update({
            f'{cls}_Pct': round(count / total * 100, 2) if total else 0
            for cls, count in result.items()
        })
        return result

    def generate_monthly_analysis(self):
        """Generates monthly trend analysis."""
        logging.info("Generating Monthly Analysis...")
        self.df['YEAR_MONTH'] = self.df['RECALL_DATE'].dt.to_period('M')
        monthly_groups = self.df.groupby('YEAR_MONTH')
        results = []
        for period, group in monthly_groups:
            stats = {
                'YearMonth': str(period), 'Year': period.year, 'Month': period.month,
                'TotalRecalls': len(group), 'UniqueBrands': group['BRAND_NAME'].nunique(),
                'UniqueProducts': group['COMMON_NAME'].nunique(),
                'DominantPathogen': self._get_mode(group['AREA_OF_CONCERN']),
                'DominantClass': self._get_mode(group['RECALL_CLASS']),
                'UniqueIncidents': group['RECALL_NUMBER'].nunique()
            }
            stats.update(self._get_severity_counts(group))
            results.append(stats)
        self.monthly_df = pd.DataFrame(results).sort_values('YearMonth')
        self.monthly_df['Rolling3M_Recalls'] = self.monthly_df['TotalRecalls'].rolling(window=3, min_periods=1).mean().round(2)
        self.monthly_df['Monthly_Change_Pct'] = self.monthly_df['TotalRecalls'].pct_change().mul(100).round(2)

    def generate_brand_analysis(self):
        """Generates analysis focused on brands."""
        logging.info("Generating Brand Analysis...")
        results = []
        for brand, group in self.df.groupby('BRAND_NAME'):
            if brand == 'unknown': continue
            first_recall = group['RECALL_DATE'].min()
            last_recall = group['RECALL_DATE'].max()
            active_days = (last_recall - first_recall).days if last_recall > first_recall else 0
            stats = {
                'BrandName': brand.title(), 'TotalRecalls': len(group),
                'UniqueProducts': group['COMMON_NAME'].nunique(), 'UniquePathogens': group['AREA_OF_CONCERN'].nunique(),
                'YearsActive': group['YEAR'].nunique(), 'DominantPathogen': self._get_mode(group['AREA_OF_CONCERN']),
                'DominantClass': self._get_mode(group['RECALL_CLASS']), 'DominantSeason': self._get_mode(group['SEASON']),
                'FirstRecall': first_recall.date(), 'LastRecall': last_recall.date(),
                'AvgDaysBetweenRecalls': round(active_days / len(group), 1) if len(group) > 1 else 0,
            }
            stats.update(self._get_severity_counts(group))
            results.append(stats)
        self.brand_df = pd.DataFrame(results).sort_values('TotalRecalls', ascending=False)

    def generate_seasonal_analysis(self):
        """Generates seasonal analysis, broken down by year."""
        logging.info("Generating Seasonal Analysis...")
        self.df['YEAR_SEASON'] = self.df['YEAR'].astype(str) + '_' + self.df['SEASON']
        results = []
        for year_season, group in self.df.groupby('YEAR_SEASON'):
            year, season = year_season.split('_')
            stats = {
                'Year': int(year), 'Season': season, 'TotalRecalls': len(group),
                'UniqueBrands': group['BRAND_NAME'].nunique(),
                'DominantPathogen': self._get_mode(group['AREA_OF_CONCERN']),
                'DominantClass': self._get_mode(group['RECALL_CLASS']),
                'UniqueIncidents': group['RECALL_NUMBER'].nunique()
            }
            stats.update(self._get_severity_counts(group))
            results.append(stats)
        self.seasonal_df = pd.DataFrame(results).sort_values(['Year', 'Season'])

    def generate_pathogen_analysis(self):
        """Generates analysis on pathogens by month and severity."""
        logging.info("Generating Pathogen Analysis...")
        pm_results = []
        for (pathogen, month), group in self.df.groupby(['AREA_OF_CONCERN', 'MONTH']):
            if pathogen == 'unknown': continue
            stats = { 'Pathogen': pathogen.title(), 'Month': month, 'TotalRecalls': len(group),
                      'UniqueBrands': group['BRAND_NAME'].nunique(), 'DominantClass': self._get_mode(group['RECALL_CLASS']), }
            stats.update(self._get_severity_counts(group))
            pm_results.append(stats)
        self.pathogen_month_df = pd.DataFrame(pm_results).sort_values(['Pathogen', 'Month'])
        ps_results = []
        total_records = len(self.df)
        pathogen_totals = self.df['AREA_OF_CONCERN'].value_counts()
        for (pathogen, severity), group in self.df.groupby(['AREA_OF_CONCERN', 'RECALL_CLASS']):
            if pathogen == 'unknown': continue
            path_total = pathogen_totals.get(pathogen, 0)
            ps_results.append({ 'Pathogen': pathogen.title(), 'Severity': severity, 'TotalRecalls': len(group),
                                'PctOfPathogen': round(len(group) / path_total * 100, 2) if path_total else 0,
                                'PctOfAllRecalls': round(len(group) / total_records * 100, 2),
                                'UniqueBrands': group['BRAND_NAME'].nunique() })
        self.pathogen_severity_df = pd.DataFrame(ps_results).sort_values(['Pathogen', 'TotalRecalls'], ascending=[True, False])

    def generate_incident_analysis(self):
        """Generates analysis based on individual recall incidents (batches)."""
        logging.info("Generating Incident Analysis...")
        incidents = self.df.groupby('RECALL_NUMBER').agg(
            StartDate=('RECALL_DATE', 'min'),
            EndDate=('RECALL_DATE', 'max'),
            NumItems=('COMMON_NAME', 'count'), # Count all rows (items)
            UniqueProducts=('COMMON_NAME', 'nunique'),
            BrandsInvolved=('BRAND_NAME', 'nunique'),
            BrandList=('BRAND_NAME', lambda x: ', '.join(sorted(x.unique()))),
            Pathogen=('AREA_OF_CONCERN', self._get_mode),
            Severity=('RECALL_CLASS', self._get_mode)
        ).reset_index()
        incidents['DurationDays'] = (incidents['EndDate'] - incidents['StartDate']).dt.days
        # Rename columns for clarity before merge (avoid conflicts)
        incidents.rename(columns={
            'Pathogen': 'Incident_Pathogen',
            'Severity': 'Incident_Severity',
            'BrandsInvolved': 'Incident_BrandsInvolved',
            'UniqueProducts': 'Incident_UniqueProducts',
            'NumItems': 'Incident_NumItems',
            'DurationDays': 'Incident_DurationDays'
        }, inplace=True)
        self.incident_df = incidents.sort_values('StartDate', ascending=False)
        logging.info(f"Generated {len(self.incident_df)} incident summaries.")


    def generate_brand_connections(self):
        """Infers potential brand connections based on co-occurrence in recalls."""
        logging.info("Generating Brand Connections...")
        incidents_with_brands = self.df.groupby('RECALL_NUMBER')['BRAND_NAME'].unique().reset_index()
        incidents_with_brands['BRAND_NAME'] = incidents_with_brands['BRAND_NAME'].apply(
            lambda x: sorted([b for b in x if b != 'unknown'])
        )
        multi_brand_incidents = incidents_with_brands[incidents_with_brands['BRAND_NAME'].apply(len) > 1]
        edge_counter = Counter()
        for brands in multi_brand_incidents['BRAND_NAME']:
            for brand1, brand2 in combinations(brands, 2):
                edge_counter[tuple(sorted((brand1, brand2)))] += 1
        if not edge_counter:
            logging.warning("No multi-brand incidents found.")
            self.brand_connections_df = pd.DataFrame(columns=['Brand1', 'Brand2', 'Weight'])
            return
        connections = [ {'Brand1': pair[0].title(), 'Brand2': pair[1].title(), 'Weight': count}
                        for pair, count in edge_counter.items() ]
        self.brand_connections_df = pd.DataFrame(connections).sort_values('Weight', ascending=False)

    def generate_ml_dataset(self):
        """ Generates an enhanced dataset with features for ML generalization. """
        logging.info("Generating Enhanced Dataset for ML...")
        self.ml_df = self.df.copy()

        # --- Time-Based Features ---
        self.ml_df['DAYS_SINCE_LAST_BRAND_RECALL'] = self.ml_df.groupby('BRAND_NAME')['RECALL_DATE'].diff().dt.days.fillna(0)
        self.ml_df['PRODUCT_KEY'] = self.ml_df['BRAND_NAME'] + '_' + self.ml_df['COMMON_NAME']
        self.ml_df['DAYS_SINCE_LAST_PROD_RECALL'] = self.ml_df.groupby('PRODUCT_KEY')['RECALL_DATE'].diff().dt.days.fillna(0)

        # --- Frequency Features ---
        self.ml_df['BRAND_RECALL_FREQ'] = self.ml_df['BRAND_NAME'].map(self.df['BRAND_NAME'].value_counts())
        self.ml_df['PATHOGEN_FREQ'] = self.ml_df['AREA_OF_CONCERN'].map(self.df['AREA_OF_CONCERN'].value_counts())

        # --- *** NEW: Merge Incident Features *** ---
        if self.incident_df is not None and not self.incident_df.empty:
            logging.info("Merging incident features into ML dataset...")
            incident_features_to_merge = [
                'RECALL_NUMBER', 'Incident_Pathogen', 'Incident_Severity',
                'Incident_BrandsInvolved', 'Incident_UniqueProducts',
                'Incident_NumItems', 'Incident_DurationDays'
            ]
            self.ml_df = pd.merge(
                self.ml_df,
                self.incident_df[incident_features_to_merge],
                on='RECALL_NUMBER',
                how='left'
            )
            # Fill NaNs that might occur if an incident wasn't processed (shouldn't happen, but safe)
            for col in incident_features_to_merge:
                if col in self.ml_df.columns and self.ml_df[col].isnull().any():
                     if pd.api.types.is_numeric_dtype(self.ml_df[col]):
                         self.ml_df[col].fillna(0, inplace=True)
                     else:
                         self.ml_df[col].fillna('N/A', inplace=True)
        else:
            logging.warning("Incident DataFrame not available, skipping merge.")


        # --- Categorical Encoding ---
        self.ml_df = pd.get_dummies(self.ml_df, columns=['SEASON', 'DAY_OF_WEEK', 'RECALL_CLASS', 'DEPTH'],
                                    prefix=['Season', 'Day', 'Class', 'Depth'],
                                    dummy_na=True, # Create a column for NA values too
                                    drop_first=False) # Keep all for clarity, can drop later if needed

        # Add numerical class if needed (useful for some models or as target)
        class_map = {'Class I': 1, 'Class II': 2, 'Class III': 3, 'Unknown': 0}
        # Use the original RECALL_CLASS before dummification
        self.ml_df['RECALL_CLASS_NUM'] = self.df['RECALL_CLASS'].map(class_map)

        # Clean up
        self.ml_df = self.ml_df.drop(columns=['PRODUCT_KEY'], errors='ignore')
        logging.info(f"ML dataset created with {len(self.ml_df.columns)} features.")

    # --- generate_summary_report & save_all_files (largely unchanged) ---
    def generate_summary_report(self, output_dir):
        """Generates a text summary of key findings."""
        logging.info("Generating Summary Report...")
        filepath = os.path.join(output_dir, 'analysis_summary.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Advanced CFIA Food Recall Analysis Summary (V3)\n")
            f.write("="*60 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.excel_file}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Records Analyzed: {len(self.df):,}\n")
            f.write(f"Date Range: {self.df['RECALL_DATE'].min().date()} to {self.df['RECALL_DATE'].max().date()}\n")
            f.write(f"Unique Brands: {self.df['BRAND_NAME'].nunique():,}\n")
            f.write(f"Unique Incidents: {self.df['RECALL_NUMBER'].nunique():,}\n")
            f.write("\n--- Recall Class Breakdown ---\n")
            class_counts = self.df['RECALL_CLASS'].value_counts()
            total = len(self.df)
            for cls, count in class_counts.items():
                f.write(f"  {cls}: {count:,} ({round(count/total*100, 1)}%)\n")

            f.write("\n--- Top 5 Brands by Recall Count ---\n")
            if hasattr(self, 'brand_df') and not self.brand_df.empty:
                for _, row in self.brand_df.head(5).iterrows():
                    f.write(f"  {row['BrandName']}: {row['TotalRecalls']:,} recalls\n")

            f.write("\n--- Insights from Incident (Batch) Analysis ---\n")
            if self.incident_df is not None and not self.incident_df.empty:
                avg_items = self.incident_df['Incident_NumItems'].mean()
                avg_brands = self.incident_df['Incident_BrandsInvolved'].mean()
                max_brands = self.incident_df['Incident_BrandsInvolved'].max()
                f.write(f"  Average items per incident: {avg_items:.1f}\n")
                f.write(f"  Average brands per incident: {avg_brands:.1f}\n")
                f.write(f"  Max brands in a single incident: {max_brands}\n")
                f.write("  * This highlights that many recalls are complex, multi-brand events.\n")

            f.write("\n--- Potential Research/ML Questions ---\n")
            f.write("  * Can incident size/complexity predict recall class or duration?\n")
            f.write("  * Do brands involved in multi-brand incidents have a higher future recall risk?\n")
            f.write("  * Which features (pathogen, season, brand history, incident size) are most predictive of Class I recalls?\n")
            f.write("="*60 + "\n")
            logging.info(f"Summary report saved to {filepath}")

    def save_all_files(self, output_dir):
        """Saves all generated DataFrames to CSV files."""
        logging.info(f"Saving output files to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        files_to_save = {
            'monthly_df': 'cfia_monthly_analysis.csv', 'brand_df': 'cfia_brand_analysis.csv',
            'seasonal_df': 'cfia_seasonal_by_year.csv', 'pathogen_month_df': 'cfia_pathogen_month.csv',
            'pathogen_severity_df': 'cfia_pathogen_severity.csv', 'incident_df': 'cfia_incident_analysis.csv',
            'brand_connections_df': 'cfia_brand_connections.csv', 'ml_df': 'cfia_enhanced_dataset_ml.csv',
        }
        for df_name, filename in files_to_save.items():
            if hasattr(self, df_name):
                df_to_save = getattr(self, df_name)
                if df_to_save is not None and not df_to_save.empty:
                    filepath = os.path.join(output_dir, filename)
                    try:
                        df_to_save.to_csv(filepath, index=False, encoding='utf-8')
                        logging.info(f"Saved {filepath}")
                    except Exception as e: logging.error(f"Failed to save {filepath}: {e}")
                else: logging.warning(f"Skipping save for '{filename}' (empty/not generated).")
        self.generate_summary_report(output_dir)

    def run_complete_analysis(self, output_dir):
        """Runs all analysis steps and saves the results."""
        if self.df is None: return
        self.generate_monthly_analysis()
        self.generate_brand_analysis()
        self.generate_seasonal_analysis()
        self.generate_pathogen_analysis()
        self.generate_incident_analysis() # Must run before ML if merging
        self.generate_brand_connections()
        self.generate_ml_dataset() # Now depends on incident_analysis
        self.save_all_files(output_dir)
        logging.info("Analysis complete.")

def main():
    """Parses arguments and runs the analysis."""
    parser = argparse.ArgumentParser(description="Advanced CFIA Food Recall Analysis Script (V3).")
    parser.add_argument("excel_file", help="Path to the input CFIA data Excel file.")
    parser.add_argument("output_directory", nargs='?', default='./cfia_analysis_output_v3',
                        help="Directory to save outputs (default: ./cfia_analysis_output_v3).")
    args = parser.parse_args()

    if not os.path.exists(args.excel_file):
        logging.error(f"Error: Input file '{args.excel_file}' not found!")
        sys.exit(1)

    analyzer = CFIAAdvancedAnalyzerV3(args.excel_file)
    analyzer.run_complete_analysis(args.output_directory)
    print("\nAnalysis complete. Outputs saved to:", args.output_directory)

if __name__ == "__main__":
    main()