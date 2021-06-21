import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow_hub as hub
import csv
import cairo
from igraph.drawing.text import TextDrawer

# 1 or more data files, exported from a CrunchBase rounds search (CSV download)
rounds_csv_file_names = ['data/tiger-rounds-6-18-2021.csv', 'data/coatue-rounds-6-14-2021.csv']  # ['data/laas-summit-list-21-76-16-2021.csv']

# uniform arrangement of the data frame
col_title = 'Title'
col_name = 'Name'
col_series = 'Series'
col_money = 'Money'
col_industries = 'Industries'
col_description = 'Description'
cols_list = [col_title, col_name, col_series, col_money, col_industries, col_description]


# data loader: df[ Title, Name, Series, Money, Industries, Description ]
def cb_load_csv_rounds(csv_file_name, csv_type):
    df = pd.read_csv(csv_file_name)
    if csv_type == 'company_rounds':
        df.rename(columns={
            "Organization Name": col_name,
            "Funding Type": col_series,
            "Money Raised Currency (in USD)": col_money,
            "Organization Industries": col_industries,
            "Organization Description": col_description,
        }, inplace=True)
    elif csv_type == 'company_list':
        df.rename(columns={
            "Organization Name": col_name,
            # Series
            "Total Funding Amount Currency (in USD)": col_money,
            "Industries": col_industries,
            "Description": col_description,
        }, inplace=True)
        df[col_series] = 'Unknown'
    else:
        raise Exception('Wrong CSV file type')
    df[col_title] = df.apply(lambda row: row[col_name] + ' (' + (str(round(row[col_money] / 1E+06)) if np.isfinite(row[col_money]) else '') + ' M)', axis=1)
    return df[cols_list]


# sentence similarity, using USE from TF-Hub (instead of Sentence-Transformers, for instance)
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def text_to_embeds_use(model, text_list):
    text_embeddings = model(text_list)
    correlation = np.inner(text_embeddings, text_embeddings)
    return 'use', text_embeddings.numpy(), correlation


# analysis from correlations matrices: group (sort rows, cols)
def plot_correlation_sorted_df(df, corr_matrix, df_labels_header, chart_title):
    def sort_df_by_correlation(_df, _corr_matrix):
        curr_i = 0
        sorted_i = [curr_i]
        while len(sorted_i) < len(_df):
            ith_corr = _corr_matrix[curr_i].copy()
            while True:
                closest_i = np.argmax(ith_corr)
                if closest_i in sorted_i:
                    ith_corr[closest_i] = -1
                    continue
                curr_i = closest_i
                break
            sorted_i.append(curr_i)
        return _df.reindex(index=sorted_i), _corr_matrix[sorted_i, :][:, sorted_i]

    def plot_corr_matrix_sns(_corr_matrix, labels, _title):
        # mask = np.zeros_like(corr)
        # mask[np.triu_indices_from(mask)] = True
        sns.set(font_scale=1.0)
        g = sns.heatmap(
            data=_corr_matrix,
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=np.max(_corr_matrix),
            # mask=mask,
            cmap="YlOrRd")
        g.set_xticklabels(labels, rotation=90)
        g.set_title(_title)

    df, corr_matrix = sort_df_by_correlation(df, corr_matrix)
    plot_corr_matrix_sns(corr_matrix, df[df_labels_header], chart_title)


# Plot a Network Graph using the correlation matrix as edge weights, and the investment size as node weights
def save_network_graph(df, corr_matrix, graph_title, graph_file_name, layout_algo='lgl', corr_threshold=0.5):
    graph = nx.Graph()
    mpl_color_map = plt.cm.get_cmap('summer')  # summer, autumn, Wistia

    # add nodes
    for v_idx, row in zip(range(len(df)), df.values):
        # row - array as per cols_list
        label = row[1]

        # style vertex by founding Stage
        frame_width = 0
        series = row[2]
        if 'Unknown' in series:
            frame_width = 1
            color = '#fff'
        elif 'Seed' in series:
            color = mpl_color_map(0)
        elif 'Series ' in series:
            char_distance = ord(series.replace('Series ', '').upper()[0]) - ord('A')
            color = mpl_color_map(np.clip(char_distance / 6, 0, 1))
        else:
            frame_width = 1
            color = '#fff'

        # style variant: unknown raise amount
        money = row[3]
        millions = round(money / 1E+06 if not np.isnan(money) else 0)
        if millions == 0:
            millions = 200
            frame_width = 1
            color = '#FF00EE80'
            label += " $?"
        else:
            # label += f"\n${millions}"
            pass

        # add the node
        size = 2 * np.sqrt(millions)
        graph.add_node(v_idx, label=label, size=size, color=color, frame_width=frame_width)

    # add edges
    for i in range(len(df) - 1):
        for j in range(i + 1, len(df)):
            weight_ij = corr_matrix[i][j]
            if weight_ij > corr_threshold:
                graph.add_edge(i, j, weight=weight_ij ** 2, color=(1 / 2, 0.5 / 2, 0, weight_ij ** 4))

    # # layout graph using weight as tension
    # print('Node layout in progress...')
    # pos = nx.fruchterman_reingold_layout(graph, k=5, weight='weight', iterations=5000, seed=1234)
    # print('...done')
    #
    # # manually draw it all
    # node_sizes = [2 * int(x) for x in nx.get_node_attributes(graph, 'value')]
    # node_labels = nx.get_node_attributes(graph, 'label')
    # edge_colors = [(1 / 4, 0.5 / 4, 0, x * x) for x in nx.get_edge_attributes(graph, 'weight').values()]
    # nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=1)
    # nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='#ff8000')
    # nx.draw_networkx_labels(graph, pos, labels=node_labels, verticalalignment='bottom')

    # iGraph: create the network from NetworkX - seems stupid but it's easier
    gg = ig.Graph.from_networkx(graph)
    # layout_algo: many, but the following are the best: 'lgl', 'graphopt', 'dh' (the latter is very slow)
    plot = ig.plot(gg, target=graph_file_name, layout=gg.layout(layout_algo), bbox=(1920, 1280), margin=50)

    ## manual drawing of the title of the graph (not done by iGraph)
    def overlay_text(surface, text, width):
        ctx = cairo.Context(surface)
        ctx.set_font_size(36)
        drawer = TextDrawer(ctx, text, halign=TextDrawer.CENTER)
        drawer.draw_at(0, 40, width=width * 2 / 3)

    overlay_text(plot.surface, graph_title, 1280)
    plot.save()
    plot.show()


# export array as CSV, with optional header row
def save_numpy_as_csv(array, csv_file_name, headers=None, delimiter='\t'):
    print(f' - saving {csv_file_name}')
    with open(csv_file_name, 'w', encoding='utf8', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter=delimiter, lineterminator='\n')
        if headers is not None:
            writer.writerow(headers)
        for f in array:
            writer.writerow(f.tolist())


# process for an individual configuration
def analyze_csv(file_name, nlp_column, export_only=False):
    # load file
    investor_name = file_name.replace('data/', '').split('-')[0].capitalize()
    print(f'Operating on {investor_name}.\n - Loading {file_name}...')
    df_cb = cb_load_csv_rounds(file_name, 'company_list' if 'list' in file_name else 'company_rounds')
    df_cb.dropna(subset=[nlp_column], inplace=True)

    # # extract all col_industries
    # all_industries = ', '.join(sum(df_cb[[col_industries]].to_numpy().tolist(), [])).split(', ')
    # all_industries.sort()
    # from itertools import groupby
    # aaa = [len(list(group)) for key, group in groupby(all_industries)]

    # compute sentence distance from the nlp column
    print(f" - NLP analysis and correlation matrix, based off '{nlp_column}'...")
    model_name, companies_embeds, companies_corr = text_to_embeds_use(use_model, list(df_cb[nlp_column]))
    print('   single min:', np.max(np.min(companies_corr, axis=1)), ' max: ', np.max(companies_corr))

    # save raw embeds for usage (for instance with https://projector.tensorflow.org/)
    tsv_base_name = f'embeds-{nlp_column}-{model_name}-{investor_name}'
    tsv_headers = [col_name, col_title, col_series, col_money, col_description, col_industries]
    save_numpy_as_csv(companies_embeds, f'{tsv_base_name}.tsv')
    save_numpy_as_csv(df_cb[tsv_headers].to_numpy(), f'{tsv_base_name}-meta.tsv', tsv_headers)
    if export_only is True:
        print(f' - Skipping Correlation Matrix, and Network Graph...\n')
        return

    # plot the correlation matrix
    print(f' - Plotting {nlp_column} correlation matrix...')
    plt.figure()
    plot_correlation_sorted_df(df_cb, companies_corr, col_title, f'{investor_name} rounds 21.H1 - by startup {nlp_column} similarity')
    # plt.savefig('test.png')

    # save a PNG file with the graph of rounds
    print(' - Generating network graph of rounds, using the LGL algo')
    rounds_sum = round(np.sum(df_cb[col_money]) / 1E+08) / 10
    graph_title = f'{investor_name} - sum of series: {rounds_sum}B'  # Jan 1 to Jun 14, 2021 -
    png_file_name = f'graph-{investor_name}.png'
    save_network_graph(df_cb, companies_corr, graph_title, png_file_name, 'lgl', 0.2 if 'coatue' in file_name else 0.3)
    print('\n')


### MAIN
for f_name in rounds_csv_file_names:
    analyze_csv(f_name, col_industries, True)

plt.show(block=True)
