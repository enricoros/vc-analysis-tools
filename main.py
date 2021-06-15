import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow_hub as hub

rounds_csv_file_names = ['data/tiger-rounds-6-9-2021.csv']

# load CB transaction CSVs
h_title = 'Title'
h_industry = 'Organization Industries'
h_desc = 'Organization Description'


def load_cb_rounds_csv(csv_file_name):
    h_name = 'Organization Name'
    h_raise = 'Money Raised Currency (in USD)'
    headers_list = [
        'Transaction Name',
        h_name,
        'Organization Name URL',
        'Funding Type',
        h_raise,
        'Announced Date',
        'Investor Names',
        'Lead Investors',
        'Pre-Money Valuation Currency (in USD)',
        h_industry,
        'CB Rank (Funding Round)',
        h_desc,
    ]
    df = pd.read_csv(csv_file_name)
    df = df[headers_list]
    df[h_title] = df.apply(lambda row: row[h_name] + ' (' + (str(round(row[h_raise] / 1E+06)) if not np.isnan(row[h_raise]) else '') + ' M)', axis=1)
    print('Total invested capital in', len(df), 'transactions:', round(np.sum(df[h_raise]) / 1E+07) / 100, ' $Bn')
    return df


df_cb = load_cb_rounds_csv(rounds_csv_file_names[0])

# text embeddings for the 'industries'
print('Encoding...')
encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def text_to_embeds(model, text_list):
    text_embeddings = model(text_list)
    correlation = np.inner(text_embeddings, text_embeddings)
    return text_embeddings, correlation


emb_industry, corr_industry = text_to_embeds(encoder_model, list(df_cb[h_industry]))
emb_taglines, corr_taglines = text_to_embeds(encoder_model, list(df_cb[h_desc]))
print('...done')


# group DF by correlation
def sort_df_by_correlation(df, corr_matrix):
    curr_i = 0
    sorted_i = [curr_i]
    while len(sorted_i) < len(df):
        ith_corr = corr_matrix[curr_i].copy()
        while True:
            closest_i = np.argmax(ith_corr)
            if closest_i in sorted_i:
                ith_corr[closest_i] = -1
                continue
            curr_i = closest_i
            break
        sorted_i.append(curr_i)
    return df.reindex(index=sorted_i), corr_matrix[sorted_i, :][:, sorted_i]


def plot_corr_matrix_sns(corr_matrix, labels, title):
    # mask = np.zeros_like(corr)
    # mask[np.triu_indices_from(mask)] = True
    sns.set(font_scale=1.0)
    g = sns.heatmap(
        data=corr_matrix,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=np.max(corr_matrix),
        # mask=mask,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=90)
    g.set_title(title)


def plot_correlation_sorted_df(df, corr_matrix, title):
    df, corr_matrix = sort_df_by_correlation(df, corr_matrix)
    plot_corr_matrix_sns(corr_matrix, df[h_title], title)


plt.subplot()
plot_correlation_sorted_df(df_cb, corr_industry, 'Tiger Global rounds 21.H1 - by startup industry similarity')

plt.figure()
plot_correlation_sorted_df(df_cb, corr_taglines, 'Tiger Global rounds 21.H1 - by startup taglines similarity')


# subplot = plt.subplot()
# plt.savefig('test.png')

def plot_network(df, corr_matrix, layout_algo='lgl', corr_threshold=0.5):
    graph = nx.Graph()
    mpl_color_map = plt.cm.get_cmap('autumn')  # summer, Wistia

    # add nodes
    for v_idx, row in zip(range(len(df)), df.values):
        label = row[1]

        # style vertex by founding Stage
        frame_width = 0
        series = row[3]
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
        raised = row[4]
        millions = round(raised / 1E+06 if not np.isnan(raised) else 0)
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

    # Also use iGraph
    gg = ig.Graph.from_networkx(graph)
    ig.plot(gg, layout=gg.layout(layout_algo), bbox=(1920, 1280), margin=50)


plot_network(df_cb, corr_industry, 'lgl', 0.2)  # good: lgl, graphopt, dh (slowest)
plot_network(df_cb, corr_taglines, 'lgl', 0.2)  # good: lgl, graphopt, dh (slowest)

plt.show(block=True)
