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
