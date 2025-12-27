import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import scipy
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy import stats

def make_df_metadata(metadata_path):
    df_metadata=pd.read_excel(metadata_path)
    df_metadata['time_per_frame']=(df_metadata.total_duration/df_metadata.total_frame_number)
    return df_metadata

def get_time_per_frame(path,df_metadata):
    filename=os.path.basename(path)[:-4]
    time_per_frame=df_metadata[df_metadata.filename==filename].time_per_frame.iloc[0]
    return time_per_frame

def make_df_intensities_background_subtraction(path,df_metadata):
    df_intensities=pd.read_csv(path)
    df_intensities.rename(columns={df_intensities.columns[0]:"frame"}, inplace = True)
    df_intensities.rename(columns={df_intensities.columns[1]:"background"}, inplace = True)
    df_intensities=subtract_background(df_intensities)
    time_per_frame=get_time_per_frame(path,df_metadata)
    df_intensities['seconds']=df_intensities.index*time_per_frame
    df_intensities = df_intensities.drop("frame", axis=1)
    return df_intensities

def subtract_background(df_intensities):
    background=df_intensities.loc[:, "background"]
    for col in range(2, len(df_intensities.columns)):
        df_intensities.iloc[:, col] = df_intensities.iloc[:, col] - background
    return df_intensities

def get_stim_start_frame(metadata):
    with open(metadata, 'r') as file:
        lines = file.readlines()
    marker=lines[-3]
    stim_start=re.search(r'(([Time(])(\d{3,5}))',marker).group(3)
    return int(stim_start)

def get_stim_start_time_w_frame(metadata_file,path,df_metadata):
    stim_frame=get_stim_start_frame(metadata_file)
    time_per_frame=get_time_per_frame(path,df_metadata)
    stim_start_time=stim_frame*time_per_frame
    return stim_start_time

def check_if_control(path):
    filename=os.path.basename(path)
    control=re.search(r'^(\d{6}_con)',filename)
    if control == None:
        control = False
    else:
        control = True
    return control

def get_stim_time_w_frame_if_not_control(path, df_metadata):
    control=check_if_control(path)
    if control == False:
        metadata_file=path[:-16]+'.txt'
        stim_start=get_stim_start_time_w_frame(metadata_file,path,df_metadata)
        return stim_start

def make_intensities_plot_seconds(df_intensities,path,df_metadata,save=False,legend=True,excluded_cells=[]):
    filename=os.path.basename(path)
    plt.figure(figsize=(10, 6))
    df_filtered = df_intensities.drop(excluded_cells, axis=1)
    df_filtered = df_filtered.drop('background', axis=1, errors='ignore')
    for column in df_filtered.columns[0:-1]:
        sns.lineplot(x=df_intensities.seconds/60, y=df_intensities[column], label=column, linewidth=1, legend=legend)

    plt.xlabel('time[min]')
    plt.ylabel('fluorescence intensity')
    plt.title(filename[:-4])
    stim_time=get_stim_time_w_frame_if_not_control(path, df_metadata)
    if stim_time != None:
        plt.axvline(stim_time/60, c = 'red')
    if save == True:
        plt.savefig('plots/'+filename[:-4]+'.pdf', bbox_inches='tight')
    plot=plt.show()
    return plot

def plot_traces_of_recording_thesis(df_intensities,file_path,df_metadata,figsize=(10,6),legend=True,excluded_cells=[],title='', ylabel='dF/F', show_stim=True):
    fig=plt.figure(figsize=figsize)
    df_filtered = df_intensities.drop(excluded_cells, axis=1)
    df_filtered = df_filtered.drop('background', axis=1, errors='ignore')
    for column in df_filtered.columns[0:-1]:
        sns.lineplot(x=df_intensities.seconds/60, y=df_intensities[column], label=column, linewidth=1.2, legend=legend)

    stim_time=get_stim_time_w_frame_if_not_control(file_path,df_metadata)
    if stim_time != None and show_stim == True:
        plt.axvline(stim_time/60, c = 'red', linewidth=2.5)
    plt.margins(x=0)
    plt.xlabel('Time [minutes]', fontsize=15, labelpad=15)
    plt.ylabel(ylabel, fontsize=15, labelpad=10)
    plt.tick_params(labelsize=12)
    if title == 'filename':
        filename=os.path.basename(file_path)
        title = filename
    plt.title(title, fontsize=15, pad=10)
    plt.tight_layout()
    return fig

def make_df_filtered_traces(filepath,df_metadata,sigma=3): # subtract background and apply gaussian filter with sigma=x
    df_intensities=make_df_intensities_background_subtraction(filepath,df_metadata)
    df_intensities.iloc[:, 1:-1] = df_intensities.iloc[:, 1:-1].apply(lambda col: gaussian_filter1d(col, sigma))
    df_intensities_filtered = df_intensities.drop('background', axis=1, errors='ignore')
    return df_intensities_filtered

def correct_baseline_to_0(df_filtered):
    for column in df_filtered.columns[:-1]:
        min_col = np.min(df_filtered[column])
        if (min_col > 0):
            df_filtered[column] = df_filtered[column] - min_col
        else:
            df_filtered[column] = df_filtered[column] + abs(min_col)
    return df_filtered

def get_dF_F_intensities(df_intensities,percentile):
    df_norm_intensities = pd.DataFrame(index=df_intensities.index)
    for column in df_intensities.columns[:-1]:
        F = df_intensities[column].values
         # Calculate baseline as nth percentile of entire trace
        F0 = np.percentile(F, percentile, method='lower')      
        # Calculate deltaF/F
        df_norm_intensities[f'{column}'] = (F - F0) / F0
    return df_norm_intensities

def make_df_intensities_normalized(path,df_metadata,percentile=5,sigma=3,normalization='dF/F'): 
    df_intensities=make_df_filtered_traces(path,df_metadata,sigma=sigma)
    if normalization == 'dF/F':
        df_norm_intensities=get_dF_F_intensities(df_intensities, percentile=percentile)
    elif normalization == 'z_score':
        df_norm_intensities=get_zscore_intensities(df_intensities)
    time_per_frame=get_time_per_frame(path,df_metadata)
    df_norm_intensities['seconds']=df_intensities.index*time_per_frame
    return df_norm_intensities

def get_zscore_intensities(df_traces):
    data_cols = df_traces.iloc[:, :-1]
    time_col = df_traces.iloc[:, -1]
    means = data_cols.mean(axis=0)
    stds = data_cols.std(axis=0)
    data_normalized = (data_cols - means) / stds
    
    # Combine normalized data with unchanged time column
    df_normalized = pd.concat([data_normalized, time_col], axis=1)
    return df_normalized

def analyze_calcium_peaks(trace, sampling_rate, params):
    # Convert time-based parameters to frames
    min_distance = int(params['min_peak_distance_sec'] * sampling_rate)
    min_width = int(params['min_width_sec'] * sampling_rate)
    max_width = int(params['max_width_sec'] * sampling_rate)
    # Find peaks
    peaks, properties = find_peaks(
        trace,
        height=params['height_threshold'],
        prominence=params['prominence_threshold'],
        distance=min_distance,
        width=(min_width, max_width),
        rel_height=0.5  # Full width at half maximum
    )
    return peaks, properties

def plot_peak_analysis(trace_file,df_metadata,params):
    df_intensities_norm=make_df_intensities_normalized(path=trace_file,df_metadata=df_metadata,normalization='z_score')
    metadata_file=trace_file[:-16]+'.txt'
    filename=os.path.basename(trace_file)
    control=check_if_control(trace_file)
    if control == False:
        stim_frame=get_stim_start_frame(metadata_file)
    else:
        pass

    peaks_list = []
    for column in df_intensities_norm.iloc[:,0:-1].columns:

        trace=df_intensities_norm.loc[:,column]
        time_per_frame=get_time_per_frame(trace_file,df_metadata)
        sampling_rate=1/time_per_frame
    
        peaks, props = analyze_calcium_peaks(trace, sampling_rate=sampling_rate,params=params)
        peak_count = len(peaks)
        peaks_list.append(peak_count)
        
        plt.figure(figsize=(12, 4))
        plt.plot(trace)
        plt.plot(peaks, trace[peaks], 'x')
        if control == False:
            plt.axvline(stim_frame, c = 'red')
        else:
            pass
        
        plt.xlabel('Time (frames)')
        plt.ylabel('Fluorescence Z-Score')
        plt.title(f'{filename}, {column}, detected {len(peaks)} peaks')
        plot=plt.show()
    return plot, peaks_list

def get_peak_frequency(trace, sampling_rate, params):
    time_per_frame=1/sampling_rate
    peaks, props = analyze_calcium_peaks(trace, sampling_rate=sampling_rate,params=params)
    peak_count = len(peaks)
    peak_frequency_per_sec = peak_count / (len(trace)*time_per_frame)
    peak_frequency_per_min = peak_frequency_per_sec*60
    return peak_frequency_per_min

def calculate_frequency_change (peak_frequency_baseline, peak_frequency_stim):
    if peak_frequency_baseline != 0:
        frequency_change=((peak_frequency_stim-peak_frequency_baseline)/peak_frequency_baseline)*100 #freq change in percent
    elif peak_frequency_baseline == 0 and peak_frequency_stim != 0:
        frequency_change='activated'
    elif peak_frequency_baseline == 0 and peak_frequency_stim == 0:
        frequency_change='inactive'
    else:
        frequency_change='check'
    return frequency_change

def extract_condition (filename):
    pattern = r'^\d{6}_([^_]+)_'
    match = re.search(pattern, filename)
    if match:
        condition = match.group(1)
        if condition == 'con':
            condition = 'control'
        elif condition == 'cocaine20uM':
            condition = 'cocaine'
        else:
            pass
    else:
        condition = 'unknown'
    return condition

def make_df_frequency_analysis_for_recording (trace_file, df_metadata, params):
    df_intensities_norm=make_df_intensities_normalized(path=trace_file,df_metadata=df_metadata,normalization='z_score')
    metadata_file=trace_file[:-16]+'.txt'
    time_per_frame=get_time_per_frame(trace_file,df_metadata)
    sampling_rate=1/time_per_frame
    
    control=check_if_control(trace_file)
    if control == False:
        stim_frame=get_stim_start_frame(metadata_file)
    else:
        stim_frame=int(len(df_intensities_norm)/2)
    
    column_names = []
    baseline_frequencies = []
    stim_frequencies = []
    frequency_changes = []
    
    for column in df_intensities_norm.iloc[:,0:-1].columns:
        trace=df_intensities_norm.loc[:,column]
        trace_before_stim=trace[:stim_frame]
        trace_after_stim=trace[stim_frame:]
        peak_frequency_baseline=get_peak_frequency(trace_before_stim,sampling_rate,params=params)
        peak_frequency_stim=get_peak_frequency(trace_after_stim,sampling_rate,params=params)
        frequency_change=calculate_frequency_change (peak_frequency_baseline, peak_frequency_stim)
    
        column_names.append(column)
        baseline_frequencies.append(peak_frequency_baseline)
        stim_frequencies.append(peak_frequency_stim)
        frequency_changes.append(frequency_change)

    duration_pre_min = (stim_frame*time_per_frame)/60
    duration_post_frames = len(df_intensities_norm) - stim_frame
    duration_post_min = (duration_post_frames*time_per_frame)/60

    df_frequency_analysis = pd.DataFrame({
        'recording': os.path.basename(trace_file),
        'trace': column_names,
        'freq_pre_stim': baseline_frequencies,
        'freq_post_stim': stim_frequencies,
        'frequency_change': frequency_changes
    })
    df_frequency_analysis['treatment'] = df_frequency_analysis['recording'].apply(extract_condition)
    df_frequency_analysis['responsivity'] = df_frequency_analysis['frequency_change'].apply(classify_responsivity)
    df_frequency_analysis['duration_pre_min'] = duration_pre_min
    df_frequency_analysis['duration_post_min'] = duration_post_min
    return df_frequency_analysis

def classify_responsivity(value):
    pct_threshold = 30
    if value == 'activated':
        return 'activated'
    elif value == 'inactive':
        return 'inactive'
    else:
        try:
            num_value = float(value)
            if num_value >= pct_threshold:
                return 'increasing'
            elif num_value <= -pct_threshold:
                return 'decreasing'
            elif -pct_threshold < num_value < pct_threshold:
                return 'invariant'
        except (ValueError, TypeError):
            return 'unknown'
    return 'unknown'

def make_df_frequency_analysis_all (analysis_files, df_metadata, params):
    dfs_all=[]
    
    for file in analysis_files:
        df_frequency_analysis=make_df_frequency_analysis_for_recording (file, df_metadata, params=params)
        dfs_all.append(df_frequency_analysis)
    
    df_frequency_analysis_all = pd.concat(dfs_all, ignore_index=True)
    return df_frequency_analysis_all


def test_frequency_changes(cell_summary, groupby='treatment', alpha=0.05, test='ttest_rel'):

    # Calculate mean frequencies per recording
    recording_means = cell_summary.groupby(['recording', groupby])[
        ['freq_pre_stim', 'freq_post_stim']
    ].mean().reset_index()
    
    results = []
    
    for group in recording_means[groupby].unique():
        group_data = recording_means[recording_means[groupby] == group]
        
        # Calculate differences
        differences = group_data['freq_post_stim'] - group_data['freq_pre_stim']
        n = len(differences)
        
        if n>=3:
            # Check normality of differences
            _, normality_p = stats.shapiro(differences)
            is_normal = normality_p > alpha
            
            if is_normal or test == 'ttest_rel':
                # Paired t-test
                stat, p_value = stats.ttest_rel(
                    group_data['freq_post_stim'], 
                    group_data['freq_pre_stim']
                )
                test_used = 'Paired t-test'
            else:
                # Wilcoxon signed-rank test
                stat, p_value = stats.wilcoxon(
                    group_data['freq_post_stim'], 
                    group_data['freq_pre_stim']
                )
                test_used = 'Wilcoxon signed-rank'
        else:
            stat, p_value = np.nan, np.nan
            test_used = 'N/A (n < 3)'
            is_normal = 'N/A (n < 3)'
        
        results.append({
            'group': group,
            'n': n,
            'mean_diff': differences.mean(),
            'normality': is_normal,
            'test_used': test_used,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    return pd.DataFrame(results)


def get_significance_stars(p_value):
    """Convert p-value to significance stars."""
    if np.isnan(p_value):
        return 'N/A'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def plot_frequency_comparison_with_stats(cell_summary, groupby='treatment', 
                                        figsize=(12, 5), error_type='sem',
                                        show_stats=True, test='ttest_rel', group_order=None, group_colors=None):

    # Calculate mean frequencies per recording
    recording_means = cell_summary.groupby(['recording', groupby])[
        ['freq_pre_stim', 'freq_post_stim']
    ].mean().reset_index()
    
    # Perform statistical tests
    if show_stats:
        stats_results = test_frequency_changes(cell_summary, groupby=groupby, test=test)
    
    # Get unique groups (treatments)
    groups = recording_means[groupby].unique()
    n_groups = len(groups)

    if group_order is not None:
        # Filter group_order to only include groups present in data
        groups = [g for g in group_order if g in groups]
    
    # Create subplots
    fig, axes = plt.subplots(1, n_groups, figsize=figsize, sharey=True)
    if n_groups == 1:
        axes = [axes]
    
    # Set up colors
    if group_colors is not None:
        # Use user-provided colors
        colors = [group_colors.get(g, 'gray') for g in groups]
    else:
        # Use default color palette
        colors = sns.color_palette("husl", n_groups)

    # Calculate unified significance marker height
    all_pre_values = recording_means['freq_pre_stim'].values
    all_post_values = recording_means['freq_post_stim'].values
    y_max_recordings = max(all_pre_values.max(), all_post_values.max())
    unified_y_sig = y_max_recordings * 1.03
    
    for idx, (group, ax) in enumerate(zip(groups, axes)):
        # Filter data for this group
        group_data = recording_means[recording_means[groupby] == group]
        
        # Calculate means and errors
        mean_pre = group_data['freq_pre_stim'].mean()
        mean_post = group_data['freq_post_stim'].mean()
        
        if error_type == 'sem':
            error_pre = group_data['freq_pre_stim'].sem()
            error_post = group_data['freq_post_stim'].sem()
        else:  # 'std'
            error_pre = group_data['freq_pre_stim'].std()
            error_post = group_data['freq_post_stim'].std()
        
        # Plot bars
        bar_positions = [0, 1]
        means = [mean_pre, mean_post]
        errors = [error_pre, error_post]
        
        bars = ax.bar(bar_positions, means, yerr=errors, 
                     alpha=0.6, color=colors[idx], 
                     capsize=5, width=0.5, edgecolor='black', linewidth=1)
        
        # Overlay individual recordings as connected points
        for _, row in group_data.iterrows():
            ax.plot([0, 1], [row['freq_pre_stim'], row['freq_post_stim']], 
                   'o-', alpha=0.5, color='gray', linewidth=1, markersize=6, zorder=3)
        
        # Add significance markers
        if show_stats:
            group_stats = stats_results[stats_results['group'] == group].iloc[0]
            p_value = group_stats['p_value']
            sig_marker = get_significance_stars(p_value)
            
            # Draw line and marker
            ax.plot([0, 1], [unified_y_sig, unified_y_sig], 'k-', linewidth=1)
            ax.text(0.5, unified_y_sig, sig_marker, ha='center', va='bottom', 
                   fontsize=14)
        
        # Formatting
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.015)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre-stim', 'Post-stim'], fontsize=15)
        ax.set_title(f'{group}', fontsize=15, pad=10)
        
        if idx == 0:
            error_label = 'SEM' if error_type == 'sem' else 'SD'
            ax.set_ylabel(f'Frequency (peaks/min)\nMean ± {error_label}', fontsize=15, labelpad=15)
    
    plt.tight_layout()
    return fig, stats_results if show_stats else None

def calculate_active_cells_per_recording_norm(cell_summary, groupby='treatment', normalize_by_time=True):

    results = []
    
    for recording in cell_summary['recording'].unique():
        recording_data = cell_summary[cell_summary['recording'] == recording]
        group = recording_data[groupby].iloc[0]
        
        n_cells_total = len(recording_data)
        n_active_pre = (recording_data['freq_pre_stim'] > 0).sum()
        n_active_post = (recording_data['freq_post_stim'] > 0).sum()
        
        prop_active_pre = n_active_pre / n_cells_total if n_cells_total > 0 else 0
        prop_active_post = n_active_post / n_cells_total if n_cells_total > 0 else 0
        
        result = {
            'recording': recording,
            'group': group,
            'n_cells_total': n_cells_total,
            'n_active_pre': n_active_pre,
            'n_active_post': n_active_post,
            'prop_active_pre': prop_active_pre,
            'prop_active_post': prop_active_post
        }
        
        if normalize_by_time:
            # Duration should be same for all cells in a recording
            duration_pre = recording_data['duration_pre_min'].iloc[0]
            duration_post = recording_data['duration_post_min'].iloc[0]
            
            prop_active_pre_per_min = prop_active_pre / duration_pre if duration_pre > 0 else 0
            prop_active_post_per_min = prop_active_post / duration_post if duration_post > 0 else 0
            
            result['prop_active_pre_per_min'] = prop_active_pre_per_min
            result['prop_active_post_per_min'] = prop_active_post_per_min
            result['duration_pre_min'] = duration_pre
            result['duration_post_min'] = duration_post
        
        results.append(result)
    
    return pd.DataFrame(results)


def test_active_cells_changes_norm(cell_summary, groupby='treatment', alpha=0.05, normalize_by_time=True, test='ttest_rel'):

    recording_summary = calculate_active_cells_per_recording_norm(cell_summary, groupby=groupby, 
                                                             normalize_by_time=normalize_by_time)
    # Choose which columns to test
    if normalize_by_time:
        pre_col = 'prop_active_pre_per_min'
        post_col = 'prop_active_post_per_min'
    else:
        pre_col = 'prop_active_pre'
        post_col = 'prop_active_post'
    
    results = []
    
    for group in recording_summary['group'].unique():
        group_data = recording_summary[recording_summary['group'] == group]
        
        # Paired test on proportions
        n = len(group_data)
        
        if n >= 3:
            # Check normality of differences
            differences = group_data[post_col] - group_data[pre_col]
            _, normality_p = stats.shapiro(differences)
            is_normal = normality_p > alpha
            
            if is_normal or test == 'ttest_rel':
                # Paired t-test
                stat, p_value = stats.ttest_rel(
                    group_data[post_col],
                    group_data[pre_col]
                )
                test_used = 'Paired t-test'
            
            else:
                # Wilcoxon signed-rank test
                stat, p_value = stats.wilcoxon(
                    group_data[post_col],
                    group_data[pre_col]
                )
                test_used = 'Wilcoxon signed-rank'
        else:
            stat, p_value = np.nan, np.nan
            test_used = 'N/A (n < 3)'
            is_normal = 'N/A (n < 3)'
        
        mean_diff = (group_data[post_col] - group_data[pre_col]).mean()
        
        results.append({
            'group': group,
            'n_recordings': n,
            'mean_diff': mean_diff,
            'normality': is_normal,
            'test_used': test_used,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    return pd.DataFrame(results)


def plot_active_cells_comparison_norm(cell_summary, groupby='treatment', 
                                 figsize=(12, 5), plot_type='proportion',
                                 show_stats=True, normalize_by_time=True, test='ttest_rel',group_order=None, group_colors=None):

    
    # Calculate per-recording active cells
    recording_summary = calculate_active_cells_per_recording_norm(cell_summary, groupby=groupby,
                                                             normalize_by_time=normalize_by_time)
    
    # Perform statistical tests if requested
    if show_stats:
        stats_results = test_active_cells_changes_norm(cell_summary, groupby=groupby,
                                                  normalize_by_time=normalize_by_time, test=test)
    
    # Get unique groups
    groups = recording_summary['group'].unique()
    n_groups = len(groups)

    if group_order is not None:
        # Filter group_order to only include groups present in data
        groups = [g for g in group_order if g in groups]
    
    # Create subplots
    fig, axes = plt.subplots(1, n_groups, figsize=figsize, sharey=True)
    if n_groups == 1:
        axes = [axes]
    
    # Set up colors
    if group_colors is not None:
        # Use user-provided colors
        colors = [group_colors.get(g, 'gray') for g in groups]
    else:
        # Use default color palette
        colors = sns.color_palette("husl", n_groups)
    
    # Determine what to plot
    if plot_type == 'proportion':
        if normalize_by_time:
            pre_col, post_col = 'prop_active_pre_per_min', 'prop_active_post_per_min'
            ylabel = 'Proportion of active cells / min'
        else:
            pre_col, post_col = 'prop_active_pre', 'prop_active_post'
            ylabel = 'Proportion of active cells'
    elif plot_type == 'percentage':
        if normalize_by_time:
            recording_summary['pct_active_pre'] = recording_summary['prop_active_pre_per_min'] * 100
            recording_summary['pct_active_post'] = recording_summary['prop_active_post_per_min'] * 100
            ylabel = 'Percentage of active cells / min (%)'
        else:
            recording_summary['pct_active_pre'] = recording_summary['prop_active_pre'] * 100
            recording_summary['pct_active_post'] = recording_summary['prop_active_post'] * 100
            ylabel = 'Percentage of active cells (%)'
        pre_col, post_col = 'pct_active_pre', 'pct_active_post'
    else:  # 'count'
        pre_col, post_col = 'n_active_pre', 'n_active_post'
        ylabel = 'Number of active cells'

    # Calculate unified significance marker height
    all_pre_values = recording_summary[pre_col].values
    all_post_values = recording_summary[post_col].values
    y_max_recordings = max(all_pre_values.max(), all_post_values.max())
    unified_y_sig = y_max_recordings * 1.05
    
    for idx, (group, ax) in enumerate(zip(groups, axes)):
        group_data = recording_summary[recording_summary['group'] == group]
        
        # Calculate means and errors
        mean_pre = group_data[pre_col].mean()
        mean_post = group_data[post_col].mean()
        error_pre = group_data[pre_col].sem()
        error_post = group_data[post_col].sem()
        
        # Plot bars
        bars = ax.bar([0, 1], [mean_pre, mean_post], 
                     yerr=[error_pre, error_post],
                     alpha=0.6, color=colors[idx], 
                     capsize=5, width=0.5, edgecolor='black', linewidth=1.5)
        
        # Overlay individual recordings
        for _, row in group_data.iterrows():
            ax.plot([0, 1], [row[pre_col], row[post_col]], 
                   'o-', alpha=0.5, color='gray', linewidth=1, markersize=6, zorder=3)
        
        # Add significance markers
        if show_stats:
            group_stats = stats_results[stats_results['group'] == group].iloc[0]
            p_value = group_stats['p_value']
            sig_marker = get_significance_stars(p_value)
                
            ax.plot([0, 1], [unified_y_sig, unified_y_sig], 'k-', linewidth=1.5)
            ax.text(0.5, unified_y_sig, sig_marker, ha='center', va='bottom', 
                       fontsize=14)

        # Formatting
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.015)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre-stim', 'Post-stim'], fontsize=15)
        ax.set_title(f'{group}', fontsize=15, pad=10)
        
        if idx == 0:
            ax.set_ylabel(f'{ylabel}\nMean ± SEM', fontsize=15, labelpad=15)
    
    plt.tight_layout()
    return fig, stats_results if show_stats else None

def make_pie_chart_responsivity(df, figsize=(12,5),pctdistance=0.8):

    order_categories = ['inactive','invariant','increasing','decreasing','activated']
    
    # Separate data by treatment condition and reindex to ensure same order
    control_data = df[df['treatment'] == 'control']['responsivity'].value_counts()
    cocaine_data = df[df['treatment'] == 'cocaine']['responsivity'].value_counts()
    control_data = control_data.reindex(order_categories, fill_value=0)
    cocaine_data = cocaine_data.reindex(order_categories, fill_value=0)
    
    colors = plt.cm.Set1(range(len(order_categories)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # don't show 0% on the plot when category is empty
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 0 else ''
    
    # Plot control pie chart
    wedges1, texts1, autotexts1 = ax1.pie(control_data.values, labels=None, autopct=autopct_format, 
                                            startangle=90, colors=colors, pctdistance=pctdistance, textprops={'fontsize': 14})
    ax1.set_title('Control ' f"({control_data.sum()} cells in total)", fontsize=15, pad=10)
    
    # Plot cocaine pie chart
    wedges2, texts2, autotexts2 = ax2.pie(cocaine_data.values, labels=None, autopct=autopct_format, 
                                            startangle=90, colors=colors, pctdistance=pctdistance, textprops={'fontsize': 14})
    ax2.set_title('Cocaine ' f"({cocaine_data.sum()} cells in total)", fontsize=15, pad=10)
    
    fig.legend(wedges1, order_categories, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=len(order_categories), fontsize=14)
    plt.tight_layout()
    return fig



