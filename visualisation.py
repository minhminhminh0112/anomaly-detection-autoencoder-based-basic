import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def compare_point_wise(real_df: pd.DataFrame, synth_df: pd.DataFrame, num_cols: list[str], right_tail:bool=True, n_cols:int = 4, figsize:tuple=None, quantile:float=None, title: str = None, save_path: str = None):
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    if figsize is None:
        figsize = (4*n_cols, 4*n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        if quantile is None:
            x_raw= real_df[col]
            x_rec = synth_df[col]
        else:
            real_data = real_df[col]
            synthetic_data = synth_df[col]
            qnt_value = real_data.quantile(quantile)
            if right_tail:
                extreme_mask = real_data >= qnt_value
            else:
                extreme_mask = real_data <= qnt_value
            mask = extreme_mask.values  # boolean mask
            x_raw = real_data[mask]
            x_rec = synthetic_data[mask]
        if 'ACCT' in col:
            # Convert to integers for ACCT columns since they represent counts
            x_raw_int = x_raw.round().astype(int)
            x_rec_int = x_rec.round().astype(int)
            axes[i].scatter(x_raw_int, x_rec_int, alpha=0.7)
            if quantile is None:
                axes[i].plot([x_raw_int.min(), x_raw_int.max()], [x_raw_int.min(), x_raw_int.max()], 'r--', lw=2)
        else:
            axes[i].scatter(x_raw,x_rec, alpha=0.7)
            axes[i].plot([x_raw.min(), x_raw.max()], [x_raw.min(), x_raw.max()], 'r--', lw=2)  # perfect match line

        axes[i].set_xlabel("Raw data")
        axes[i].set_ylabel("Reconstructed data")
        axes[i].set_title(col, size = 10)
    if quantile is None:
        fig.suptitle(title if title else 'Point-wise Comparison of Raw and Reconstructed Data')
    else:
        fig.suptitle(title if title else f'Extreme Values Plot ({'right' if right_tail else 'left'} {round(1- quantile, 2)} tail)')
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    if save_path is not None:
        plt.savefig(save_path + f'extreme_values_{int((1-quantile)*100)}_right_tail.png')
    plt.show()


def compare_distribution_plot(col_names:dict, X, pred_df: pd.DataFrame, title:str = None, save_path:str = None):
    """ Compare the distribution of input data and predicted data for each column.
    Args:col_names should have the following keys ['cat:list, 'num': list, 'bool': list, 'date': list]"""
    types= ['cat','bool','num','date']
    ordered_cols = []
    for i in types:
        if i in col_names.keys():
            ordered_cols += col_names[i]
    n_cols = 4
    n_rows = int(np.ceil(len(ordered_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    
    for col, ax in zip(ordered_cols,axes.flatten()):
        if 'num' in col_names.keys() and (col in col_names['num'] or col == 'DISBURSAL_DATE'):
            df_compare = pd.concat([X[col], pred_df[col]], axis=1, join="outer")
            ax.hist(df_compare, 12, density=1, histtype='bar', stacked=False, label = ['input','predicted'], color=['blue', 'red'], alpha=0.6)
        else:
            col_unique_counts = X[col].value_counts()
            predicted_unique_count = pred_df[col].value_counts()
            if col in col_names['bool']:
                col_unique_counts.index = col_unique_counts.index.astype(int)
                predicted_unique_count.index = predicted_unique_count.index.astype(int)
            else:
                col_unique_counts.index = col_unique_counts.index.astype(str)
                predicted_unique_count.index = predicted_unique_count.index.astype(str)
            df_compare = pd.concat([col_unique_counts, predicted_unique_count], axis=1, join="outer")
            X_axis = np.arange(len(X[col].unique()))
            ax.bar(X_axis-0.2, df_compare.iloc[:,0], width=0.4, label = 'input', color='blue', alpha=0.6)
            ax.bar(X_axis+0.2, df_compare.iloc[:,1], width = 0.4, label = 'predicted', color='red', alpha=0.6)
            ax.set_xticks(X_axis, df_compare.index, rotation = 45, ha= 'right')
            # plt.setp(axes[i].get_xticklabels(), ha='right')
        ax.set_xlabel(col)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    if title is not None:
        fig.suptitle(title, size = 16, y=1.0) #{model_path} 
    if save_path is not None:
        plt.savefig(save_path + 'distributions.png')
    fig.tight_layout() #rect=[0, 0, 1, 0.99]
    fig.show()

def recon_error_scatter(recon_errors, y, threshold_value):
    fig, ax = plt.subplots(figsize=(14, 7))

    normal_mask = y == 0
    anomaly_mask = y == 1
    ax.scatter(recon_errors.index[normal_mask], 
            recon_errors[normal_mask],
            alpha=0.5,
            s=50,
            zorder = 2,
            label=f'Normal ({normal_mask.sum()})')

    ax.scatter(recon_errors.index[anomaly_mask], 
            recon_errors[anomaly_mask],
            alpha=0.7,
            s=70,
            zorder = 1,
            label=f'Anomaly ({anomaly_mask.sum()})')
    ax.axhline(y=threshold_value, color='orange', linestyle='--', 
           linewidth=2.5, label='Threshold at top true n', alpha=0.9, zorder=5)
    ax.set_xlabel('Sample Index', fontsize=13)
    ax.set_ylabel('Reconstruction Error', fontsize=13)
    ax.set_title('Reconstruction Error from Autoencoder: Normal vs Anomaly', 
                fontsize=15, pad=20)
    ax.set_facecolor('#f7f7f7')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    plt.tight_layout()
    plt.show()

def recon_error_hist(recon_errors, y):
    fig, ax = plt.subplots(figsize=(14, 7))

    normal_mask = y == 0
    anomaly_mask = y == 1
    ax.hist( 
        recon_errors[normal_mask],
        alpha=0.9,
        zorder = 1,
        bins=30,
        label=f'Normal ({normal_mask.sum()})')

    ax.hist(
        recon_errors[anomaly_mask],
        alpha=0.9,
        zorder = 2,
        bins=30,
        label=f'Anomaly ({anomaly_mask.sum()})')

    ax.set_xlabel('Reconstruction Error', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Reconstruction Error from Autoencoder: Normal vs Anomaly', 
            fontsize=15, pad=20)
    ax.set_facecolor('#f7f7f7')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    plt.tight_layout()
    plt.show()

def evaluation_metrics_over_percentile_thresholds(recon_errors, y):
    from sklearn.metrics import f1_score, precision_score, recall_score

    percentiles = np.arange(0, 100, 1)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    thresholds = []

    for p in percentiles:
        threshold = np.percentile(recon_errors, p)
        thresholds.append(threshold)
        
        y_pred = (recon_errors > threshold).astype(int)
        
        f1_scores.append(f1_score(y, y_pred))
        precision_scores.append(precision_score(y, y_pred, zero_division=0))
        recall_scores.append(recall_score(y, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(percentiles, f1_scores, color='blue', 
            linewidth=2.5, label='F1-Score', alpha=0.8)
    ax.plot(percentiles, precision_scores,color='green', 
            linewidth=2,  label='Precision', alpha=0.7)
    ax.plot(percentiles, recall_scores, color='orange', 
            linewidth=2, label='Recall', alpha=0.7)


    ax.set_xlabel('Percentile Threshold', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Classification Metrics vs Percentile Threshold', 
                fontsize=15, pad=20)

    # ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f7f7f7')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.set_xlim(percentiles.min() - 2, percentiles.max() + 2)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()

def feature_importance_plot(feature_importance_sorted:pd.Series):
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(feature_importance_sorted.index, 
                feature_importance_sorted.values,
                alpha=0.8,
                linewidth=0.5)

    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance Ranking', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
    ax.set_facecolor('#f7f7f7')

    plt.tight_layout()
    plt.show()

def feature_importance_groups_plot(x1:pd.Series, x2:pd.Series, x3:pd.Series):
    sorted_index = x1.sort_values().index
    n = len(sorted_index)
    x = np.arange(n)

    n_bars = 3
    total_height = 0.8               
    bar_height = total_height / n_bars

    offsets = np.linspace(-total_height/2 + bar_height/2,
                        total_height/2 - bar_height/2,
                        n_bars)

    fig, ax = plt.subplots(figsize=(10, 6))

    ys1 = x + offsets[0]
    ys2 = x + offsets[1]
    ys3 = x + offsets[2]

    bars1 = ax.barh(ys1,
                    x1[sorted_index].values,
                    height=bar_height, label='All Importance',
                    color='blue', alpha=0.8, linewidth=0.5)

    bars2 = ax.barh(ys2,
                    x2[sorted_index].values,
                    height=bar_height, label='Normal Loan Importance',
                    color='green', alpha=0.8, linewidth=0.5)

    bars3 = ax.barh(ys3,
                    x3[sorted_index].values,
                    height=bar_height, label='Default Loan Importance',
                    color='orange', alpha=0.8, linewidth=0.5)

    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance Ranking', pad=12)

    ax.set_yticks(x)
    ax.set_yticklabels(sorted_index)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
    ax.set_facecolor('#f7f7f7')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

def reconstruction_error_one_sample(error:pd.Series):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(error.index, error.values, alpha=0.8)
    ax.set_ylabel("Reconstruction Error")
    ax.set_xlabel("Features")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_xticklabels(error.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()