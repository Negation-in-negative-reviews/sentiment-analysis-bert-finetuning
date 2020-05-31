import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
LINEWIDTH = 2
sns.set(font_scale=2.0, rc={
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": 20,
        "ps.useafm": True,
        "font.sans-serif": ["Helvetica"],
        "pdf.use14corefonts": True,
        # "text.usetex": True,
    })
sns.set_style("white")
filetype = ".png"

mpl.rcParams['hatch.linewidth'] = 1.0
mpl.rcParams['hatch.color'] = "black"

def draw_grouped_barplot(data_df, colors, x_label: str, y_label: str, seaborn_hue: str, bargraph_savepath: str, 
    title: str="", y_axis_name:str="", ylim_top: float=None):      
    print(bargraph_savepath)    
    plt.figure()
    plt.figure(figsize=(20,8))
    data_df = data_df.sort_values(by=[x_label])

    # data_df = pd.DataFrame({
    #     x_label: [d[x_label] for d in data],
    #     y_label: [d[y_label] for d in data],
    #     "sem_value": [d["sem_value"] for d in data],
    #     seaborn_hue: [d[seaborn_hue] for d in data]
    # })
    
    print(data_df)
    # colors = [(84/255, 141/255, 255/255),  (84/255, 141/255, 255/255)]*2

    subx = data_df[seaborn_hue].unique()
    u = data_df[x_label].unique()
    x = np.arange(len(u))
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = data_df[data_df[seaborn_hue] == gr]
        plt.bar(x+offsets[i], dfg[y_label].values, width=width,
                # yerr=dfg["sem_value"].values, 
                # error_kw={"elinewidth": 10},
                color=colors[i])
    plt.xticks(x, u)
    ax = plt.gca()
    # ax = sns.barplot(x=x_label, y=y_label, data=data_df, hue=seaborn_hue,
        # palette=colors)
    if ylim_top != None:
        ax.set_ylim(0, ylim_top)
    ax.get_yaxis().get_offset_text().set_x(-0.05)
    ax.set_xlabel('')
    ax.set_ylabel(y_axis_name, labelpad=30)

    for idx, thisbar in enumerate(ax.patches):
        if idx>=len(ax.patches)/2:
            thisbar.set_hatch('/')
            thisbar.set_edgecolor('black')

    pos_rev_patch = mpl.patches.Patch(facecolor=colors[0], alpha=1, edgecolor='black', hatch='', label='Positive reviews')
    neg_rev_patch = mpl.patches.Patch(facecolor=colors[1], alpha=1, edgecolor='black', hatch='/', label='Negative reviews')
    plt.legend(handles=[pos_rev_patch, neg_rev_patch], loc='upper left')    

    ax.tick_params(axis='x', which='major', pad=35)
    ax.tick_params(axis='y', which='major', pad=25)
    
    plt.tight_layout()
    plt.savefig(bargraph_savepath+filetype, dpi=600)


def draw_grouped_barplot_three_subbars(data_df, colors, x_label: str, y_label: str, seaborn_hue: str, bargraph_savepath: str, 
    title: str="", y_axis_name:str="", ylim_top: float=None):      
    print(bargraph_savepath)    
    plt.figure()
    plt.figure(figsize=(20,8))
    data_df = data_df.sort_values(by=[x_label])
    patterns = ['', '/', '-']
    # data_df = pd.DataFrame({
    #     x_label: [d[x_label] for d in data],
    #     y_label: [d[y_label] for d in data],
    #     "sem_value": [d["sem_value"] for d in data],
    #     seaborn_hue: [d[seaborn_hue] for d in data]
    # })
    
    print(data_df)
    # colors = [(84/255, 141/255, 255/255),  (84/255, 141/255, 255/255)]*2

    subx = data_df[seaborn_hue].unique()
    # print("subx: ", len(subx))
    u = data_df[x_label].unique()
    x = np.arange(len(u))
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width = np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = data_df[data_df[seaborn_hue] == gr]
        plt.bar(x+offsets[i], dfg[y_label].values, width=width,
                # yerr=dfg["sem_value"].values, 
                # error_kw={"elinewidth": 10},
                hatch=patterns[i], 
                color=colors[i])
    plt.xticks(x, u)
    ax = plt.gca()
    # ax = sns.barplot(x=x_label, y=y_label, data=data_df, hue=seaborn_hue,
        # palette=colors)
    if ylim_top != None:
        ax.set_ylim(0, ylim_top)
    ax.get_yaxis().get_offset_text().set_x(-0.05)
    ax.set_xlabel('')
    ax.set_ylabel(y_axis_name, labelpad=30)

    # for idx, thisbar in enumerate(ax.patches):
    #     if idx>=len(ax.patches)/2:
    #         thisbar.set_hatch('/')
    #         thisbar.set_edgecolor('black')

    # pos_rev_patch = mpl.patches.Patch(facecolor=colors[0], alpha=1, edgecolor='black', hatch='', label='Positive reviews')
    # neg_rev_patch = mpl.patches.Patch(facecolor=colors[1], alpha=1, edgecolor='black', hatch='/', label='Negative reviews')
    # plt.legend(handles=[pos_rev_patch, neg_rev_patch], loc='upper left')    

    ax.tick_params(axis='x', which='major', pad=35)
    ax.tick_params(axis='y', which='major', pad=25)
    plt.legend(data_df[seaborn_hue].unique())
    
    plt.tight_layout()
    plt.savefig(bargraph_savepath+filetype, dpi=600)



