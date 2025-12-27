import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import scipy
from math import floor
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy import sparse
from scipy import stats

# extract info from file path or metadata

def extract_raw_filename(filepath):
    pattern = r'(\d{6}_.*?_region\d+)'
    match = re.search(pattern, filepath)
    if match:
        raw_filename = match.group(1)
    return raw_filename

def extract_framerate(filepath):
    pattern = r'(\d{1}_\d{1,2}Hz)'
    match = re.search(pattern, filepath)
    if match:
        framerate_str = match.group(1)
        framerate = float(framerate_str.replace('Hz', '').replace('_', '.'))
    return framerate

def get_stim_start_frame(metadata_folder,filepath):
    raw_filename=extract_raw_filename(filepath)
    metadata_file=metadata_folder+raw_filename+'.txt'
    with open(metadata_file, 'r') as file:
        lines = file.readlines()
    marker=lines[-3]
    stim_start=re.search(r'(([Time(])(\d{3,5}))',marker).group(3)
    stim_start=int(stim_start)
    stim_start_corrected=correct_stim_frame_if_downsampled(filepath,stim_start)
    return stim_start_corrected

def get_stim_start_time(metadata_folder,filepath):
    framerate=extract_framerate(filepath)
    stim_start_frame=get_stim_start_frame(metadata_folder,filepath)
    stim_start_time=stim_start_frame/framerate
    return stim_start_time

def extract_treatments(filepath):
    pattern = r'\d{6}_([a-z]+)'
    match = re.search(pattern, filepath)
    if match:
        treatment = match.group(1)
    else:
        treatment = 'unknown'
    if treatment == 'glutamate':
        pretreatment = os.path.basename(os.path.dirname(filepath))
    else:
        pretreatment = None
    return treatment, pretreatment

# making dataframes for traces

def make_df_caiman_traces(filepath):
    df_trace = pd.read_csv(filepath,header=0)
    framerate = extract_framerate(filepath)
    df_trace['seconds']=df_trace.index/framerate
    return df_trace

def correct_baseline_to_0(df_filtered):
    for column in df_filtered.columns[:-1]:
        min_col = np.min(df_filtered[column])
        if (min_col > 0):
            df_filtered[column] = df_filtered[column] - min_col
        else:
            df_filtered[column] = df_filtered[column] + abs(min_col)
    return df_filtered

def make_df_filtered_traces(filepath,sigma=3):
    df_trace = make_df_caiman_traces(filepath)
    df_trace.iloc[:, :-1] = df_trace.iloc[:, :-1].apply(lambda col: gaussian_filter1d(col, sigma))
    df_trace_corrected = correct_baseline_to_0(df_trace)
    return df_trace_corrected

def trim_long_traces(df,filepath,metadata_folder):
    stim_time = get_stim_start_time(metadata_folder,filepath)
    duration_post_stim = df.seconds.iloc[-1] - stim_time
    if duration_post_stim > 540:
        cutoff_index = (df.seconds > (stim_time + 540)).idxmax()
        df_trimmed = df.loc[:cutoff_index]
    else:
        df_trimmed = df.copy()
    return df_trimmed

def make_df_filtered_and_trimmed_traces(filepath,metadata_folder,sigma=3):
    df = make_df_filtered_traces(filepath,sigma)
    df_trimmed = trim_long_traces(df,filepath,metadata_folder)
    return df_trimmed

def make_df_zscore(df_traces):
    data_cols = df_traces.iloc[:, :-1]
    time_col = df_traces.iloc[:, -1]
    means = data_cols.mean(axis=0)
    stds = data_cols.std(axis=0)
    data_normalized = (data_cols - means) / stds
    
    # Combine normalized data with unchanged time column
    df_normalized = pd.concat([data_normalized, time_col], axis=1)
    return df_normalized

# get and analyze component areas

def get_IDs_accepted_cells(filepath):
    df_trace = pd.read_csv(filepath,header=0)
    accepted_components=df_trace.columns
    cell_IDs=[]
    for comp in accepted_components:
        cell_ID=comp.split()[1]
        cell_IDs.append(int(cell_ID))
    return cell_IDs

def get_npz_file(filepath):
    filename=os.path.basename(filepath)[:-4]
    npz_filename='SpatialInfo_'+filename+'.npz'
    folderpath=os.path.dirname(filepath)
    npz_file_path=folderpath+'/'+npz_filename
    return npz_file_path

def get_areas_of_components(filepath):
    npz_file=get_npz_file(filepath)
    A = sparse.load_npz(npz_file)
    areas = np.asarray((A > 0).sum(axis=0)).flatten() #gives you the spatial footprint of a cell in number of pixels
    return areas

#others

def correct_stim_frame_if_downsampled(filepath,stim_start):
    framerate=extract_framerate(filepath)
    if framerate == 2.65:
        stim_start_correct = floor(stim_start/2)
    else:
        stim_start_correct = stim_start
    return stim_start_correct

# peak detection and analysis

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

def plot_peak_analysis(filepath,metadata_folder,params):
    df_filtered_traces = make_df_filtered_and_trimmed_traces(filepath,metadata_folder,sigma=3)
    df_zscore_traces = make_df_zscore(df_filtered_traces)
    framerate = extract_framerate(filepath)
    filename = os.path.basename(filepath)

    for column in df_zscore_traces.iloc[:,:-1].columns:
        trace = df_zscore_traces.loc[:,column]
        peaks, props = analyze_calcium_peaks(trace, sampling_rate=framerate, params=params)
        peak_count = len(peaks)
        stim_start = get_stim_start_frame(metadata_folder,filepath)
    
        plt.figure(figsize=(12, 4))
        plt.plot(trace)
        plt.plot(peaks, trace[peaks], 'x')
        plt.axvline(stim_start, c = 'red')
        plt.xlabel('Time (frames)')
        plt.ylabel('Fluorescence Z-Score')
        plt.title(f'{filename}, {column}, detected {len(peaks)} peaks')
        plot=plt.show()
    return plot

def make_df_peak_analysis_for_recording(filepath,metadata_folder,params):
    df_filtered_traces = make_df_filtered_and_trimmed_traces(filepath,metadata_folder,sigma=3)
    df_zscore_traces = make_df_zscore(df_filtered_traces)
    framerate = extract_framerate(filepath)
    filename = os.path.basename(filepath)
    cell_areas = get_areas_of_components(filepath)
    
    all_peaks = []
    
    for column in df_zscore_traces.iloc[:,:-1].columns:
        trace = df_zscore_traces.loc[:,column]
        peaks, properties = analyze_calcium_peaks(trace,sampling_rate=framerate,params=params)
        
        # Create dataframe for this cell's peaks
        if len(peaks) > 0:  # Only if peaks were found
            cell_df = pd.DataFrame({
                'cell_ID': int(column.split()[1]),
                'peak_index': range(len(peaks)),
                'peak_frame': peaks,
                'prominence': properties['prominences'],
                'height': properties['peak_heights'],
                'width': properties['widths'],
                'width_sec': properties['widths']/framerate
            })
            all_peaks.append(cell_df)
            
    # Combine all cells into one dataframe and add columns based on the recording
    df = pd.concat(all_peaks, ignore_index=True)
    df.insert(0,'recording', filename)
    df['treatment'] = extract_treatments(filepath)[0]
    df['pretreatment'] = extract_treatments(filepath)[1]
    df['stim_frame'] = get_stim_start_frame(metadata_folder,filepath)
    df['cell_area'] = [cell_areas[i] for i in df['cell_ID']]
    df['framerate'] = framerate
    df['last_frame'] = len(df_zscore_traces)-1
    return df

def make_df_peak_analysis_all (filepaths,metadata_folder,params):
    dfs_all=[]
    
    for file in filepaths:
        try:
            df_peak_analysis=make_df_peak_analysis_for_recording(file,metadata_folder,params)
            dfs_all.append(df_peak_analysis)
        except:
            print('check file: '+file)
    
    df_peak_analysis_all = pd.concat(dfs_all, ignore_index=True)
    return df_peak_analysis_all

def make_df_peak_frequencies_per_cell(peak_df):
    # Group by recording and cell
    grouped = peak_df.groupby(['recording', 'cell_ID'])
    
    summary_list = []
    
    for (recording, cell_id), cell_peaks in grouped:
        # Get cell-level properties (same for all peaks of this cell)
        stim_frame = cell_peaks['stim_frame'].iloc[0]
        treatment = cell_peaks['treatment'].iloc[0]
        pretreatment = cell_peaks['pretreatment'].iloc[0]
        cell_area = cell_peaks['cell_area'].iloc[0]
        frame_rate = cell_peaks['framerate'].iloc[0]
        recording_duration = cell_peaks['last_frame'].iloc[0]
        
        # Separate pre and post stimulation peaks
        pre_peaks = cell_peaks[cell_peaks['peak_frame'] < stim_frame]
        post_peaks = cell_peaks[cell_peaks['peak_frame'] >= stim_frame]
        
        n_peaks_pre = len(pre_peaks)
        n_peaks_post = len(post_peaks)
        
        # Calculate duration in minutes
        duration_pre_frames = stim_frame
        duration_pre_min = duration_pre_frames / (frame_rate * 60)
        
        duration_post_frames = recording_duration - stim_frame
        duration_post_min = duration_post_frames / (frame_rate * 60)
        
        # Calculate frequencies (peaks per minute)
        freq_pre_stim = n_peaks_pre / duration_pre_min 
        freq_post_stim = n_peaks_post / duration_post_min 
        
        summary_list.append({
            'recording': recording,
            'cell_ID': cell_id,
            'freq_pre_stim': freq_pre_stim,
            'freq_post_stim': freq_post_stim,
            'n_peaks_pre': n_peaks_pre,
            'n_peaks_post': n_peaks_post,
            'duration_pre_min': duration_pre_min,
            'duration_post_min': duration_post_min,
            'treatment': treatment,
            'pretreatment': pretreatment,
            'stim_frame': stim_frame,
            'last_frame': recording_duration,
            'cell_area': cell_area,
            'frame_rate': frame_rate
        })
    
    return pd.DataFrame(summary_list)


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
                                        show_stats=True, test='ttest_rel', group_order=None, group_colors=None, title_prefix=None):
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
        ax.set_title(f'{title_prefix} {group}', fontsize=15, pad=10)
        
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
            normality = 'N/A (n < 3)'
        
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
            
            if not np.isnan(p_value):
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

def plot_traces_of_recording_thesis(df_traces,file_path,metadata_folder,figsize=(10,6),legend=True,excluded_cells=[],title=''):
    fig = plt.figure(figsize=figsize)
    df_filtered = df_traces.drop(excluded_cells, axis=1)
    for column in df_filtered.columns[:-1]:
        sns.lineplot(x=df_traces.seconds/60, y=df_traces[column], label=column, linewidth=1.2, legend=legend)
    stim_start=get_stim_start_time(metadata_folder,file_path)
    plt.axvline(stim_start/60, c = 'red', linewidth=2)
    plt.margins(x=0)
    plt.xlabel('Time [minutes]', fontsize=15, labelpad=15)
    plt.ylabel('z-score', fontsize=15, labelpad=10)
    plt.tick_params(labelsize=12)
    if title == 'filename':
        filename=os.path.basename(file_path)
        title = filename
    plt.title(title, fontsize=15, pad=10)
    plt.tight_layout()
    return fig


#unused
def get_peak_frequency(trace, sampling_rate, params):
    time_per_frame = 1/sampling_rate
    peaks, props = analyze_calcium_peaks(trace, sampling_rate=sampling_rate, params=params)
    peak_count = len(peaks)
    peak_frequency_per_sec = peak_count / (len(trace)*time_per_frame)
    peak_frequency_per_min = peak_frequency_per_sec*60
    return peak_frequency_per_min
