import pandas as pd
import os
import sys
from matplotlib import  pyplot as plt
is_windows = hasattr(sys, 'getwindowsversion')
plt.rcParams["figure.figsize"] = (20, 10)

# PATH DEFINITION
if is_windows:
    PATH_TO_DNN_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\DNN\\results\\"
else:
    PATH_TO_DNN_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/DNN/results/"

df_results=pd.read_csv(PATH_TO_DNN_RESULTS+"DNN_results.csv",sep=";")
df_results['home_id']=[cfg[0] for cfg in df_results['config'].str.split('_').to_list()]
df_results['config2']=['_'.join(cfg[1:]) for cfg in df_results['config'].str.split('_').to_list()]
home_list=df_results['home_id'].unique()
for home_id in home_list:
    df_results.sort_values(['config2'], ascending=False,inplace=True)
    df_plot=df_results[df_results['home_id']==home_id]
    df_plot.sort_values(by=['test_mse'],inplace=True)
    ax=df_plot.plot.bar(x='config2',y='test_mse')
    plt.title(home_id)
    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.7)
    # fig.canvas.manager.full_screen_toggle()
    result_png_name = PATH_TO_DNN_RESULTS + home_id + "_barplot.png"
    fig.savefig(result_png_name)
    plt.close()

    # plt.show()
    print(home_id)