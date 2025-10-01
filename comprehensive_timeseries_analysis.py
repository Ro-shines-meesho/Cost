#!/usr/bin/env python3
"""
COMPREHENSIVE TIME-SERIES Kubernetes Resource Analysis

This script analyzes the complete historical data to identify:
1. Temporal resource usage patterns
2. Application behavior over time
3. True bad actors (consistently problematic vs temporary spikes)
4. Trend analysis and seasonal patterns
5. Historical context for better decision making

Key improvements:
- Preserves all time-series data for pattern analysis
- Identifies consistent vs. temporary issues
- Analyzes resource usage trends over time
- Better bad actor detection using historical behavior
- Seasonal and cyclical pattern identification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in resource usage"""
    print("\n🕒 Analyzing temporal patterns...")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['date'] = df['timestamp'].dt.date
    
    # Calculate time-based statistics
    temporal_stats = {
        'total_timestamps': df['timestamp'].nunique(),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600,
        'avg_interval_minutes': df.groupby('node')['timestamp'].diff().dt.total_seconds().mean() / 60
    }
    
    print(f"📊 Temporal Analysis:")
    print(f"  Total timestamps: {temporal_stats['total_timestamps']}")
    print(f"  Date range: {temporal_stats['date_range']}")
    print(f"  Duration: {temporal_stats['duration_hours']:.1f} hours")
    print(f"  Average interval: {temporal_stats['avg_interval_minutes']:.1f} minutes")
    
    return temporal_stats

def calculate_node_timeseries_metrics(df):
    """Calculate comprehensive time-series metrics for each node"""
    print("\n📈 Calculating node time-series metrics...")
    
    # CRITICAL FIX: Group by node and timestamp for time-series analysis
    # This ensures we sum pod requests PER TIMESTAMP, not across all timestamps
    node_timeseries = df.groupby(['node', 'timestamp']).agg({
        'pod_cpu_req_cores': 'sum',           # Sum pod CPU requests per node per timestamp
        'pod_mem_req_GB': 'sum',              # Sum pod memory requests per node per timestamp
        'node_cpu_allocatable': 'first',      # Node capacity (same for all pods/timestamps)
        'node_mem_allocatable_GB': 'first',   # Node capacity (same for all pods/timestamps)
        'node_cpu_capacity_cores': 'first',   # Node capacity (same for all pods/timestamps)
        'node_mem_capacity_GB': 'first',      # Node capacity (same for all pods/timestamps)
        'nodepool': 'first',                  # Nodepool (same for all pods/timestamps)
        'pod': 'count'                        # Number of pods on this node at this timestamp
    }).reset_index()
    
    node_timeseries.rename(columns={'pod': 'pod_count'}, inplace=True)
    
    # Calculate utilization percentages for each timestamp (should be ≤100% in normal cases)
    node_timeseries['cpu_utilization_pct'] = (node_timeseries['pod_cpu_req_cores'] / node_timeseries['node_cpu_allocatable']) * 100
    node_timeseries['mem_utilization_pct'] = (node_timeseries['pod_mem_req_GB'] / node_timeseries['node_mem_allocatable_GB']) * 100
    
    # Calculate waste and over-allocation for each timestamp
    node_timeseries['cpu_unutilized'] = np.maximum(0, node_timeseries['node_cpu_allocatable'] - node_timeseries['pod_cpu_req_cores'])
    node_timeseries['mem_unutilized_GB'] = np.maximum(0, node_timeseries['node_mem_allocatable_GB'] - node_timeseries['pod_mem_req_GB'])
    node_timeseries['cpu_over_requested'] = np.maximum(0, node_timeseries['pod_cpu_req_cores'] - node_timeseries['node_cpu_allocatable'])
    node_timeseries['mem_over_requested_GB'] = np.maximum(0, node_timeseries['pod_mem_req_GB'] - node_timeseries['node_mem_allocatable_GB'])
    
    # Validation check for reasonable utilization
    max_cpu_util = node_timeseries['cpu_utilization_pct'].max()
    max_mem_util = node_timeseries['mem_utilization_pct'].max()
    
    print(f"✅ Generated {len(node_timeseries)} node-timestamp records")
    print(f"📊 Utilization validation:")
    print(f"   Max CPU utilization: {max_cpu_util:.1f}%")
    print(f"   Max Memory utilization: {max_mem_util:.1f}%")
    
    if max_cpu_util > 500 or max_mem_util > 500:
        print("⚠️  WARNING: Extremely high utilization detected - possible data aggregation issue!")
    
    return node_timeseries

def calculate_historical_node_summary(node_timeseries):
    """Calculate comprehensive historical statistics for each node"""
    print("\n📊 Calculating historical node summaries...")
    
    # Aggregate historical statistics per node
    node_historical = node_timeseries.groupby('node').agg({
        # Current state (latest values)
        'pod_cpu_req_cores': 'last',
        'pod_mem_req_GB': 'last',
        'node_cpu_allocatable': 'first',
        'node_mem_allocatable_GB': 'first',
        'node_cpu_capacity_cores': 'first',
        'node_mem_capacity_GB': 'first',
        'nodepool': 'first',
        'pod_count': 'last',
        
        # Historical statistics
        'cpu_utilization_pct': ['mean', 'std', 'min', 'max', 'last'],
        'mem_utilization_pct': ['mean', 'std', 'min', 'max', 'last'],
        'cpu_unutilized': ['mean', 'std', 'min', 'max', 'last'],
        'mem_unutilized_GB': ['mean', 'std', 'min', 'max', 'last'],
        'cpu_over_requested': ['mean', 'std', 'min', 'max', 'last'],
        'mem_over_requested_GB': ['mean', 'std', 'min', 'max', 'last'],
        
        # Timestamp info
        'timestamp': ['count', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    node_historical.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in node_historical.columns]
    
    # Rename for clarity
    rename_map = {
        'cpu_utilization_pct_mean': 'cpu_util_avg',
        'cpu_utilization_pct_std': 'cpu_util_std',
        'cpu_utilization_pct_min': 'cpu_util_min',
        'cpu_utilization_pct_max': 'cpu_util_max',
        'cpu_utilization_pct_last': 'cpu_util_current',
        'mem_utilization_pct_mean': 'mem_util_avg',
        'mem_utilization_pct_std': 'mem_util_std',
        'mem_utilization_pct_min': 'mem_util_min',
        'mem_utilization_pct_max': 'mem_util_max',
        'mem_utilization_pct_last': 'mem_util_current',
        'cpu_unutilized_mean': 'cpu_waste_avg',
        'cpu_unutilized_std': 'cpu_waste_std',
        'cpu_unutilized_min': 'cpu_waste_min',
        'cpu_unutilized_max': 'cpu_waste_max',
        'cpu_unutilized_last': 'cpu_waste_current',
        'mem_unutilized_GB_mean': 'mem_waste_avg',
        'mem_unutilized_GB_std': 'mem_waste_std',
        'mem_unutilized_GB_min': 'mem_waste_min',
        'mem_unutilized_GB_max': 'mem_waste_max',
        'mem_unutilized_GB_last': 'mem_waste_current',
        'cpu_over_requested_mean': 'cpu_over_avg',
        'cpu_over_requested_std': 'cpu_over_std',
        'cpu_over_requested_max': 'cpu_over_max',
        'cpu_over_requested_last': 'cpu_over_current',
        'mem_over_requested_GB_mean': 'mem_over_avg',
        'mem_over_requested_GB_std': 'mem_over_std',
        'mem_over_requested_GB_max': 'mem_over_max',
        'mem_over_requested_GB_last': 'mem_over_current',
        'timestamp_count': 'observation_count',
        'timestamp_min': 'first_seen',
        'timestamp_max': 'last_seen'
    }
    
    node_historical.rename(columns=rename_map, inplace=True)
    
    # Calculate volatility and consistency metrics
    node_historical['cpu_volatility'] = node_historical['cpu_util_std'] / (node_historical['cpu_util_avg'] + 0.1)
    node_historical['mem_volatility'] = node_historical['mem_util_std'] / (node_historical['mem_util_avg'] + 0.1)
    
    # Calculate consistency scores (lower is more consistent)
    node_historical['cpu_consistency_score'] = (node_historical['cpu_util_max'] - node_historical['cpu_util_min']) / (node_historical['cpu_util_avg'] + 0.1)
    node_historical['mem_consistency_score'] = (node_historical['mem_util_max'] - node_historical['mem_util_min']) / (node_historical['mem_util_avg'] + 0.1)
    
    # Calculate trend indicators (current vs average)
    node_historical['cpu_trend'] = node_historical['cpu_util_current'] - node_historical['cpu_util_avg']
    node_historical['mem_trend'] = node_historical['mem_util_current'] - node_historical['mem_util_avg']
    
    print(f"✅ Generated historical summaries for {len(node_historical)} nodes")
    return node_historical

def identify_temporal_bad_actors(node_historical, node_timeseries):
    """Identify bad actors using historical patterns and consistency"""
    print("\n🔍 Identifying temporal bad actors...")
    
    # Features for temporal anomaly detection
    temporal_features = node_historical[[
        'cpu_util_avg', 'cpu_util_std', 'cpu_util_max',
        'mem_util_avg', 'mem_util_std', 'mem_util_max',
        'cpu_waste_avg', 'mem_waste_avg',
        'cpu_over_avg', 'mem_over_avg',
        'cpu_volatility', 'mem_volatility',
        'cpu_consistency_score', 'mem_consistency_score',
        'observation_count'
    ]].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Standardize features for anomaly detection
    scaler = StandardScaler()
    temporal_features_scaled = scaler.fit_transform(temporal_features)
    
    # Run Isolation Forest with temporal features
    iso_forest_temporal = IsolationForest(
        n_estimators=300,
        contamination=0.15,
        random_state=42
    )
    
    node_historical['temporal_anomaly'] = iso_forest_temporal.fit_predict(temporal_features_scaled)
    node_historical['temporal_anomaly_score'] = iso_forest_temporal.decision_function(temporal_features_scaled)
    
    # Classify bad actors by type
    def classify_bad_actor_type(row):
        if row['temporal_anomaly'] == 1:
            return 'Good Actor'
        
        # Analyze the type of bad behavior
        issues = []
        
        if row['cpu_over_avg'] > 0 or row['mem_over_avg'] > 0:
            issues.append('Chronic Over-allocation')
        
        if row['cpu_waste_avg'] > 10 or row['mem_waste_avg'] > 20:
            issues.append('High Waste')
        
        if row['cpu_volatility'] > 2 or row['mem_volatility'] > 2:
            issues.append('High Volatility')
        
        if row['cpu_consistency_score'] > 5 or row['mem_consistency_score'] > 5:
            issues.append('Inconsistent Usage')
        
        if row['cpu_util_max'] > 150 or row['mem_util_max'] > 150:
            issues.append('Extreme Spikes')
        
        if not issues:
            issues.append('General Anomaly')
        
        return ' + '.join(issues)
    
    node_historical['bad_actor_type'] = node_historical.apply(classify_bad_actor_type, axis=1)
    
    # Identify different categories of problematic nodes
    temporal_bad_actors = node_historical[node_historical['temporal_anomaly'] == -1].copy()
    chronic_over_allocated = node_historical[node_historical['cpu_over_avg'] > 0].copy()
    high_waste_nodes = node_historical[
        (node_historical['cpu_waste_avg'] > 10) | 
        (node_historical['mem_waste_avg'] > 20)
    ].copy()
    volatile_nodes = node_historical[
        (node_historical['cpu_volatility'] > 1.5) | 
        (node_historical['mem_volatility'] > 1.5)
    ].copy()
    
    print(f"🚨 Temporal Bad Actor Analysis:")
    print(f"  Total bad actors: {len(temporal_bad_actors)} ({len(temporal_bad_actors)/len(node_historical)*100:.1f}%)")
    print(f"  Chronic over-allocated: {len(chronic_over_allocated)} nodes")
    print(f"  High waste nodes: {len(high_waste_nodes)} nodes")
    print(f"  Volatile nodes: {len(volatile_nodes)} nodes")
    
    return {
        'temporal_bad_actors': temporal_bad_actors,
        'chronic_over_allocated': chronic_over_allocated,
        'high_waste_nodes': high_waste_nodes,
        'volatile_nodes': volatile_nodes
    }

def analyze_pod_temporal_patterns(df):
    """Analyze pod-level temporal patterns"""
    print("\n📦 Analyzing pod temporal patterns...")
    
    # Pod time-series analysis
    pod_timeseries = df.groupby(['pod', 'timestamp']).agg({
        'pod_cpu_req_cores': 'first',
        'pod_mem_req_GB': 'first',
        'node_cpu_capacity_cores': 'first',
        'node_mem_capacity_GB': 'first',
        'namespace': 'first',
        'nodepool': 'first',
        'node': 'first'
    }).reset_index()
    
    # Calculate pod historical statistics
    pod_historical = pod_timeseries.groupby('pod').agg({
        'pod_cpu_req_cores': ['mean', 'std', 'min', 'max', 'last'],
        'pod_mem_req_GB': ['mean', 'std', 'min', 'max', 'last'],
        'node_cpu_capacity_cores': 'first',
        'node_mem_capacity_GB': 'first',
        'namespace': 'first',
        'nodepool': 'first',
        'node': 'first',
        'timestamp': 'count'
    }).reset_index()
    
    # Flatten column names
    pod_historical.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in pod_historical.columns]
    
    # Calculate pod ratios and patterns
    pod_historical['cpu_mem_ratio_avg'] = pod_historical['pod_cpu_req_cores_mean'] / pod_historical['pod_mem_req_GB_mean']
    
    # Fix column name issue - use the flattened column name
    if 'node_cpu_capacity_cores_first' in pod_historical.columns:
        pod_historical['node_cpu_mem_ratio'] = pod_historical['node_cpu_capacity_cores_first'] / pod_historical['node_mem_capacity_GB_first']
    else:
        pod_historical['node_cpu_mem_ratio'] = pod_historical['node_cpu_capacity_cores'] / pod_historical['node_mem_capacity_GB']
    pod_historical['ratio_deviation'] = pod_historical['cpu_mem_ratio_avg'] / pod_historical['node_cpu_mem_ratio']
    
    # Calculate resource stability
    pod_historical['cpu_stability'] = 1 - (pod_historical['pod_cpu_req_cores_std'] / (pod_historical['pod_cpu_req_cores_mean'] + 0.001))
    pod_historical['mem_stability'] = 1 - (pod_historical['pod_mem_req_GB_std'] / (pod_historical['pod_mem_req_GB_mean'] + 0.001))
    
    # Classify pod patterns
    def classify_pod_temporal_pattern(row):
        deviation = row['ratio_deviation']
        cpu_stability = row['cpu_stability']
        mem_stability = row['mem_stability']
        
        pattern = []
        
        # Resource ratio classification
        if pd.isna(deviation):
            pattern.append('Unknown Ratio')
        elif deviation > 2.0:
            pattern.append('Severely CPU-heavy')
        elif deviation > 1.5:
            pattern.append('CPU-heavy')
        elif deviation < 0.33:
            pattern.append('Severely Memory-heavy')
        elif deviation < 0.67:
            pattern.append('Memory-heavy')
        else:
            pattern.append('Balanced')
        
        # Stability classification
        if cpu_stability < 0.8 or mem_stability < 0.8:
            pattern.append('Unstable')
        elif cpu_stability > 0.95 and mem_stability > 0.95:
            pattern.append('Very Stable')
        else:
            pattern.append('Stable')
        
        return ' + '.join(pattern)
    
    pod_historical['temporal_pattern'] = pod_historical.apply(classify_pod_temporal_pattern, axis=1)
    
    # Calculate potential waste based on historical patterns
    def calculate_temporal_pod_waste(row):
        node_ratio = row['node_cpu_mem_ratio']
        pod_cpu = row['pod_cpu_req_cores_mean']
        pod_mem = row['pod_mem_req_GB_mean']
        
        if 'CPU-heavy' in row['temporal_pattern']:
            optimal_cpu = pod_mem * node_ratio
            cpu_waste = max(0, pod_cpu - optimal_cpu)
            mem_waste = 0
        elif 'Memory-heavy' in row['temporal_pattern']:
            optimal_mem = pod_cpu / node_ratio
            mem_waste = max(0, pod_mem - optimal_mem)
            cpu_waste = 0
        else:
            cpu_waste = 0
            mem_waste = 0
        
        return pd.Series([cpu_waste, mem_waste])
    
    pod_historical[['pod_cpu_waste_potential', 'pod_mem_waste_potential']] = pod_historical.apply(calculate_temporal_pod_waste, axis=1)
    
    print(f"✅ Analyzed {len(pod_historical)} pods with temporal patterns")
    return pod_historical

def generate_temporal_insights(node_historical, node_timeseries, temporal_stats):
    """Generate insights from temporal analysis"""
    print("\n💡 Generating temporal insights...")
    
    insights = {
        'cluster_overview': {
            'total_nodes': len(node_historical),
            'observation_period_hours': temporal_stats['duration_hours'],
            'total_observations': len(node_timeseries),
            'avg_observations_per_node': node_historical['observation_count'].mean()
        },
        'resource_trends': {
            'nodes_with_increasing_cpu': len(node_historical[node_historical['cpu_trend'] > 10]),
            'nodes_with_decreasing_cpu': len(node_historical[node_historical['cpu_trend'] < -10]),
            'nodes_with_increasing_mem': len(node_historical[node_historical['mem_trend'] > 10]),
            'nodes_with_decreasing_mem': len(node_historical[node_historical['mem_trend'] < -10])
        },
        'stability_metrics': {
            'highly_volatile_cpu_nodes': len(node_historical[node_historical['cpu_volatility'] > 2]),
            'highly_volatile_mem_nodes': len(node_historical[node_historical['mem_volatility'] > 2]),
            'consistent_nodes': len(node_historical[
                (node_historical['cpu_consistency_score'] < 1) & 
                (node_historical['mem_consistency_score'] < 1)
            ])
        }
    }
    
    return insights

def main():
    print("=" * 80)
    print("COMPREHENSIVE TIME-SERIES KUBERNETES RESOURCE ANALYSIS")
    print("=" * 80)
    
    # ============================================================================
    # STEP 1: LOAD AND PREPROCESS DATA (PRESERVE TIME-SERIES)
    # ============================================================================
    
    print("\n🔄 Loading complete time-series data...")
    df = pd.read_csv("df_cleaned_exclude_daemonset_converted.csv")
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Unique nodes: {df['node'].nunique()}")
    print(f"Unique pods: {df['pod'].nunique()}")
    print(f"Unique timestamps: {df['timestamp'].nunique()}")
    
    # Use pre-converted GB columns (no conversion needed)
    df['pod_mem_req_GB'] = df['pod_mem_req_gb']
    df['node_mem_allocatable_GB'] = df['node_memory_allocatable_gb']
    df['node_mem_capacity_GB'] = df['node_memory_capacity_gb']
    
    print("✅ Using pre-converted GB columns - no unit conversion needed")
    
    # Analyze temporal patterns
    temporal_stats = analyze_temporal_patterns(df)
    
    # ============================================================================
    # STEP 2: TIME-SERIES NODE ANALYSIS
    # ============================================================================
    
    # Calculate node metrics for each timestamp
    node_timeseries = calculate_node_timeseries_metrics(df)
    
    # Calculate comprehensive historical statistics
    node_historical = calculate_historical_node_summary(node_timeseries)
    
    # ============================================================================
    # STEP 3: TEMPORAL BAD ACTOR DETECTION
    # ============================================================================
    
    bad_actor_results = identify_temporal_bad_actors(node_historical, node_timeseries)
    
    # ============================================================================
    # STEP 4: POD TEMPORAL ANALYSIS
    # ============================================================================
    
    pod_historical = analyze_pod_temporal_patterns(df)
    
    # ============================================================================
    # STEP 5: GENERATE INSIGHTS
    # ============================================================================
    
    insights = generate_temporal_insights(node_historical, node_timeseries, temporal_stats)
    
    # ============================================================================
    # STEP 6: SAVE COMPREHENSIVE REPORTS
    # ============================================================================
    
    print("\n💾 Saving comprehensive temporal reports...")
    
    # Time-series data
    node_timeseries.to_csv('node_timeseries_analysis.csv', index=False)
    
    # Historical summaries
    node_historical.to_csv('node_historical_analysis.csv', index=False)
    pod_historical.to_csv('pod_historical_analysis.csv', index=False)
    
    # Bad actor reports
    bad_actor_results['temporal_bad_actors'].to_csv('temporal_bad_actors.csv', index=False)
    bad_actor_results['chronic_over_allocated'].to_csv('chronic_over_allocated_nodes.csv', index=False)
    bad_actor_results['high_waste_nodes'].to_csv('historical_high_waste_nodes.csv', index=False)
    bad_actor_results['volatile_nodes'].to_csv('volatile_nodes.csv', index=False)
    
    # Current state reports (for dashboard compatibility)
    # Check actual column names and map them correctly
    print("Available columns:", list(node_historical.columns))
    
    current_state = pd.DataFrame()
    current_state['node'] = node_historical['node']
    current_state['pod_cpu_req_cores'] = node_historical['pod_cpu_req_cores_last']
    current_state['pod_mem_req_GB'] = node_historical['pod_mem_req_GB_last']
    current_state['node_cpu_allocatable'] = node_historical['node_cpu_allocatable_first']
    current_state['node_mem_allocatable_GB'] = node_historical['node_mem_allocatable_GB_first']
    current_state['node_cpu_capacity_cores'] = node_historical['node_cpu_capacity_cores_first']
    current_state['node_mem_capacity_GB'] = node_historical['node_mem_capacity_GB_first']
    current_state['nodepool'] = node_historical['nodepool_first']
    current_state['pod_count'] = node_historical['pod_count_last']
    current_state['cpu_utilization_pct'] = node_historical['cpu_util_current']
    current_state['mem_utilization_pct'] = node_historical['mem_util_current']
    current_state['cpu_unutilized'] = node_historical['cpu_waste_current']
    current_state['mem_unutilized_GB'] = node_historical['mem_waste_current']
    current_state['cpu_over_requested'] = node_historical['cpu_over_current']
    current_state['mem_over_requested_GB'] = node_historical['mem_over_current']
    
    # Add utilization patterns for dashboard compatibility
    def classify_current_utilization_pattern(row):
        cpu_util = row['cpu_utilization_pct']
        mem_util = row['mem_utilization_pct']
        
        if cpu_util > 100 and mem_util > 100:
            return "Both Over-utilized"
        elif cpu_util > 100 and mem_util < 80:
            return "CPU Over-utilized, Memory Wasted"
        elif mem_util > 100 and cpu_util < 80:
            return "Memory Over-utilized, CPU Wasted"
        elif cpu_util > 80 and mem_util > 80:
            return "Both Highly Utilized"
        elif cpu_util > 80 and mem_util < 80:
            return "CPU Utilized, Memory Wasted"
        elif mem_util > 80 and cpu_util < 80:
            return "Memory Utilized, CPU Wasted"
        elif cpu_util < 80 and mem_util < 80:
            return "Both Under-utilized"
        else:
            return "Mixed Utilization"
    
    current_state['utilization_pattern'] = current_state.apply(classify_current_utilization_pattern, axis=1)
    
    # Add temporal metrics for enhanced insights
    current_state['cpu_volatility'] = node_historical['cpu_volatility']
    current_state['mem_volatility'] = node_historical['mem_volatility']
    current_state['cpu_trend'] = node_historical['cpu_trend']
    current_state['mem_trend'] = node_historical['mem_trend']
    current_state['bad_actor_type'] = node_historical['bad_actor_type']
    current_state['observation_count'] = node_historical['observation_count']
    
    current_state.to_csv('node_analysis_corrected_fixed.csv', index=False)
    
    # Generate other dashboard-compatible files
    over_utilized = current_state[
        (current_state['cpu_utilization_pct'] > 100) | 
        (current_state['mem_utilization_pct'] > 100)
    ]
    over_utilized.to_csv('over_utilized_nodes_fixed.csv', index=False)
    
    imbalanced = current_state[
        ((current_state['cpu_utilization_pct'] > 80) & (current_state['mem_utilization_pct'] < 50)) |
        ((current_state['mem_utilization_pct'] > 80) & (current_state['cpu_utilization_pct'] < 50))
    ]
    imbalanced.to_csv('imbalanced_nodes_fixed.csv', index=False)
    
    waste_nodes = current_state[
        (current_state['cpu_unutilized'] > 5) | 
        (current_state['mem_unutilized_GB'] > 10)
    ]
    waste_nodes.to_csv('waste_nodes_fixed.csv', index=False)
    
    # Save temporal bad actors as main bad actors file
    bad_actor_results['temporal_bad_actors'].to_csv('bad_actors_comprehensive.csv', index=False)
    
    # Pod analysis files - map columns correctly
    pod_current = pd.DataFrame()
    pod_current['pod'] = pod_historical['pod']
    pod_current['namespace'] = pod_historical['namespace_first']
    pod_current['nodepool'] = pod_historical['nodepool_first']
    pod_current['pod_cpu_req_cores'] = pod_historical['pod_cpu_req_cores_last']
    pod_current['pod_mem_req_GB'] = pod_historical['pod_mem_req_GB_last']
    pod_current['pod_cpu_waste'] = pod_historical['pod_cpu_waste_potential']
    pod_current['pod_mem_waste_GB'] = pod_historical['pod_mem_waste_potential']
    pod_current['resource_pattern'] = pod_historical['temporal_pattern']
    
    pod_current.to_csv('pod_resource_mismatch_analysis.csv', index=False)
    
    cpu_heavy_pods = pod_current[pod_current['resource_pattern'].str.contains('CPU-heavy', na=False)]
    cpu_heavy_pods.to_csv('cpu_heavy_pods.csv', index=False)
    
    mem_heavy_pods = pod_current[pod_current['resource_pattern'].str.contains('Memory-heavy', na=False)]
    mem_heavy_pods.to_csv('memory_heavy_pods.csv', index=False)
    
    # Generate nodepool and namespace summaries
    nodepool_summary = node_historical.groupby('nodepool_first').agg({
        'cpu_waste_avg': 'sum',
        'mem_waste_avg': 'sum',
        'cpu_over_avg': 'sum',
        'mem_over_avg': 'sum',
        'cpu_util_avg': 'mean',
        'mem_util_avg': 'mean',
        'node': 'count'
    }).reset_index()
    nodepool_summary.rename(columns={'node': 'node_count', 'nodepool_first': 'nodepool'}, inplace=True)
    nodepool_summary.to_csv('nodepool_comprehensive_analysis.csv', index=False)
    
    namespace_summary = pod_historical.groupby('namespace_first').agg({
        'pod_cpu_req_cores_mean': 'sum',
        'pod_mem_req_GB_mean': 'sum',
        'pod_cpu_waste_potential': 'sum',
        'pod_mem_waste_potential': 'sum',
        'pod': 'count'
    }).reset_index()
    namespace_summary.rename(columns={'pod': 'pod_count', 'namespace_first': 'namespace'}, inplace=True)
    namespace_summary.to_csv('namespace_comprehensive_analysis.csv', index=False)
    
    # ============================================================================
    # STEP 7: COMPREHENSIVE SUMMARY
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\n📊 TEMPORAL OVERVIEW:")
    print(f"  Observation period: {temporal_stats['duration_hours']:.1f} hours")
    print(f"  Total timestamps: {temporal_stats['total_timestamps']}")
    print(f"  Total observations: {len(node_timeseries):,}")
    print(f"  Average observations per node: {insights['cluster_overview']['avg_observations_per_node']:.1f}")
    
    print(f"\n📈 RESOURCE TRENDS:")
    print(f"  Nodes with increasing CPU usage: {insights['resource_trends']['nodes_with_increasing_cpu']}")
    print(f"  Nodes with decreasing CPU usage: {insights['resource_trends']['nodes_with_decreasing_cpu']}")
    print(f"  Nodes with increasing memory usage: {insights['resource_trends']['nodes_with_increasing_mem']}")
    print(f"  Nodes with decreasing memory usage: {insights['resource_trends']['nodes_with_decreasing_mem']}")
    
    print(f"\n🔄 STABILITY METRICS:")
    print(f"  Highly volatile CPU nodes: {insights['stability_metrics']['highly_volatile_cpu_nodes']}")
    print(f"  Highly volatile memory nodes: {insights['stability_metrics']['highly_volatile_mem_nodes']}")
    print(f"  Consistent nodes: {insights['stability_metrics']['consistent_nodes']}")
    
    print(f"\n🚨 TEMPORAL BAD ACTORS:")
    print(f"  Total temporal bad actors: {len(bad_actor_results['temporal_bad_actors'])} ({len(bad_actor_results['temporal_bad_actors'])/len(node_historical)*100:.1f}%)")
    print(f"  Chronic over-allocated nodes: {len(bad_actor_results['chronic_over_allocated'])}")
    print(f"  Historical high waste nodes: {len(bad_actor_results['high_waste_nodes'])}")
    print(f"  Volatile nodes: {len(bad_actor_results['volatile_nodes'])}")
    
    print(f"\n📦 POD TEMPORAL PATTERNS:")
    pattern_counts = pod_historical['temporal_pattern'].value_counts()
    for pattern, count in pattern_counts.head(5).items():
        print(f"  {pattern}: {count:,} pods ({count/len(pod_historical)*100:.1f}%)")
    
    print(f"\n📋 TEMPORAL REPORTS GENERATED:")
    temporal_files = [
        'node_timeseries_analysis.csv',
        'node_historical_analysis.csv',
        'pod_historical_analysis.csv',
        'temporal_bad_actors.csv',
        'chronic_over_allocated_nodes.csv',
        'historical_high_waste_nodes.csv',
        'volatile_nodes.csv'
    ]
    
    dashboard_files = [
        'node_analysis_corrected_fixed.csv',
        'over_utilized_nodes_fixed.csv',
        'imbalanced_nodes_fixed.csv',
        'waste_nodes_fixed.csv',
        'bad_actors_comprehensive.csv',
        'pod_resource_mismatch_analysis.csv',
        'cpu_heavy_pods.csv',
        'memory_heavy_pods.csv',
        'nodepool_comprehensive_analysis.csv',
        'namespace_comprehensive_analysis.csv'
    ]
    
    print("\n  📊 Temporal Analysis Files:")
    for file in temporal_files:
        print(f"    ✅ {file}")
    
    print("\n  🖥️  Dashboard Compatible Files:")
    for file in dashboard_files:
        print(f"    ✅ {file}")
    
    print(f"\n🎯 KEY TEMPORAL INSIGHTS:")
    print(f"  • Historical context preserved with {temporal_stats['total_timestamps']} timestamps")
    print(f"  • Bad actors identified using temporal patterns, not just snapshots")
    print(f"  • Resource trends and volatility patterns analyzed")
    print(f"  • Consistent vs. temporary issues differentiated")
    print(f"  • Application behavior patterns captured over time")
    
    print("\n✅ COMPREHENSIVE TEMPORAL ANALYSIS COMPLETE!")
    print("🔧 All historical patterns preserved and analyzed")
    print("📊 Dashboard compatible with enhanced temporal insights")
    print("🎯 True bad actors identified using historical behavior")

if __name__ == "__main__":
    main()
