import os
import pandas as pd
import matplotlib.pyplot as plt
from common.matplotlib_extend import plot_ma
from constants import EPOCH, FINAL_KPIS, MEAN_KPIS, PLOT_AVG_POINTS


def post_epoch_func(log_dir, n):
    progress_csv = os.path.join(log_dir, 'progress.csv')
    df = pd.read_csv(progress_csv)
    final_kpis = FINAL_KPIS
    mean_kpis = MEAN_KPIS

    for kpi in mean_kpis:
        series = map(lambda s: df[f'{s}/env_infos/{kpi} Mean'], ['evaluation'])
        plot_ma(series=series, lables=['evaluation'], title=kpi, n=n)
        plt.savefig(os.path.join(log_dir, f'plot_evaluation_{kpi}_Mean.png'))
        plt.close()

        series = map(lambda s: df[f'{s}/env_infos/{kpi} Mean'], ['exploration'])
        plot_ma(series=series, lables=['exploration'], title=kpi, n=n)
        plt.savefig(os.path.join(log_dir, f'plot_exploration_{kpi}_Mean.png'))
        plt.close()



    for kpi in final_kpis:
        series = map(lambda s: df[f'{s}/env_infos/final/{kpi} Mean'], ['evaluation'])
        plot_ma(series=series, lables=['evaluation'], title=kpi, n=n)
        plt.savefig(os.path.join(log_dir, f'plot_evaluation_{kpi}_Final.png'))
        plt.close()

        series = map(lambda s: df[f'{s}/env_infos/final/{kpi} Mean'], ['exploration'])
        plot_ma(series=series, lables=['exploration'], title=kpi, n=n)
        plt.savefig(os.path.join(log_dir, f'plot_exploration_{kpi}_Final.png'))
        plt.close()

if __name__ == "__main__":
    LOG_DIR = ""
    post_epoch_func(LOG_DIR, PLOT_AVG_POINTS)
