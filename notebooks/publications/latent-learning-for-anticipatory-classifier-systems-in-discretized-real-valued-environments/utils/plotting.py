import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pylab import cm

# Common color palette
palette = cm.get_cmap('Paired', 7)

COLORS = {
    'acs': palette(0),
    'acs2': palette(1),
    'acs2_oiq': palette(2),
    'acs2_ga': palette(3),
    'acs2_ga_oiq': palette(4),
    'yacs': palette(5),
    'dynaq': palette(6),
}


def plot_comparison(metrics):
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))
    mark_every = 10

    metrics['knowledge_100'] = metrics['knowledge'] * 100
    metrics['generalization_100'] = metrics['generalization'] * 100

    # Population
    metrics.loc['acs']['population'].plot(label='ACS', color=COLORS['acs'], marker='x', markevery=mark_every, ax=axs[0, 0])
    metrics.loc['acs2']['population'].plot(label='ACS2', color=COLORS['acs2'], marker='v', markevery=mark_every+1, ax=axs[0, 0])
    metrics.loc['acs2_oiq']['population'].plot(label='ACS2_OIQ', color=COLORS['acs2_oiq'], marker='v', markevery=mark_every+2, ax=axs[0, 0])
    metrics.loc['acs2_ga']['population'].plot(label='ACS2_GA', color=COLORS['acs2_ga'], marker='s', markevery=mark_every+3, ax=axs[0, 0])
    metrics.loc['acs2_ga_oiq']['population'].plot(label='ACS2_GA_OIQ', color=COLORS['acs2_ga_oiq'], marker='s', markevery=mark_every+4, ax=axs[0, 0])
    metrics.loc['yacs']['population'].plot(label='YACS', color=COLORS['yacs'], marker='o', markevery=mark_every+5, ax=axs[0, 0])
    metrics.loc['dynaq']['population'].plot(label='DynaQ', color=COLORS['dynaq'], marker='D', markevery=mark_every+6, ax=axs[0, 0])
    axs[0, 0].set_title('Population size')
    axs[0, 0].set_ylabel('Number of rules/classifiers')
#     axs[0, 0].legend(loc='best', frameon=False)

    # Knowledge
    axs[0, 1].set_title('Knowledge')
    metrics.loc['acs']['knowledge_100'].plot(label='ACS', color=COLORS['acs'], marker='x', markevery=mark_every, ax=axs[0, 1])
    metrics.loc['acs2']['knowledge_100'].plot(label='ACS2', color=COLORS['acs2'], marker='v', markevery=mark_every+1, ax=axs[0, 1])
    metrics.loc['acs2_oiq']['knowledge_100'].plot(label='ACS2_OIQ', color=COLORS['acs2_oiq'], marker='v', markevery=mark_every+2, ax=axs[0, 1])
    metrics.loc['acs2_ga']['knowledge_100'].plot(label='ACS2_GA', color=COLORS['acs2_ga'], marker='s', markevery=mark_every+3, ax=axs[0, 1])
    metrics.loc['acs2_ga_oiq']['knowledge_100'].plot(label='ACS2_GA_OIQ', color=COLORS['acs2_ga_oiq'], marker='s', markevery=mark_every+4, ax=axs[0, 1])
    metrics.loc['yacs']['knowledge_100'].plot(label='YACS', color=COLORS['yacs'], marker='o', markevery=mark_every+5, ax=axs[0, 1])
    metrics.loc['dynaq']['knowledge_100'].plot(label='DynaQ', color=COLORS['dynaq'], marker='D', markevery=mark_every+6, ax=axs[0, 1])
#     axs[0, 1].legend(loc='lower right', frameon=False)
    axs[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter())

    # Generalization
    axs[1, 0].set_title('Generalization')
    metrics.loc['acs']['generalization_100'].plot(label='ACS', color=COLORS['acs'], marker='x', markevery=mark_every, ax=axs[1, 0])
    metrics.loc['acs2']['generalization_100'].plot(label='ACS2', color=COLORS['acs2'], marker='v', markevery=mark_every+1, ax=axs[1, 0])
    metrics.loc['acs2_oiq']['generalization_100'].plot(label='ACS2_OIQ', color=COLORS['acs2_oiq'], marker='v', markevery=mark_every+2, ax=axs[1, 0])
    metrics.loc['acs2_ga']['generalization_100'].plot(label='ACS2_GA', color=COLORS['acs2_ga'], marker='s', markevery=mark_every+3, ax=axs[1, 0])
    metrics.loc['acs2_ga_oiq']['generalization_100'].plot(label='ACS2_GA_OIQ', color=COLORS['acs2_ga_oiq'], marker='s', markevery=mark_every+4, ax=axs[1, 0])
    metrics.loc['yacs']['generalization_100'].plot(label='YACS', color=COLORS['yacs'], marker='o', markevery=mark_every+5, ax=axs[1, 0])
    metrics.loc['dynaq']['generalization_100'].plot(label='DynaQ', color=COLORS['dynaq'], marker='D', markevery=mark_every+6, ax=axs[1, 0])
#     axs[1, 0].legend(loc='best', frameon=False)
    axs[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter())

    # Trial time
    times = metrics.groupby('agent')['time'].mean().to_dict()

    labels = ['ACS', 'ACS2', 'ACS2_OIQ', 'ACS2_GA', 'ACS2_GA_OIQ', 'YACS', 'DynaQ']
    values = [times['acs'], times['acs2'], times['acs2_oiq'], times['acs2_ga'], times['acs2_ga_oiq'], times['yacs'], times['dynaq']]
    colors = [COLORS['acs'], COLORS['acs2'], COLORS['acs2_oiq'], COLORS['acs2_ga'], COLORS['acs2_ga_oiq'], COLORS['yacs'], COLORS['dynaq']]

    axs[1, 1].bar(labels, values, color=colors)
    axs[1, 1].set_xticklabels(labels, rotation=60)
    axs[1, 1].set_title('Average trial time')
    axs[1, 1].set_ylabel('Seconds [s]')

    # create some space below the plots by increasing the bottom-value
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.16)

    # Global legend
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(values), loc='lower center', prop={'size': 23})
