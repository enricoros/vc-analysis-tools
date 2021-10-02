#
# This file only contains misc functions that were tried, but we decided to not use/proceed in that direction.
# Aka: random snippets of code
#
# import heatmap as hmz
# import matplotlib.pyplot as plt
# import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub

test_sentences = [
    '3D Technology, Aerospace, Artificial Intelligence, Manufacturing, Supply Chain Management',
    'Big Data, Data Visualization, Logistics, Software',
    'Artificial Intelligence, CRM, Information Technology, Machine Learning, Sales, Software',
    'Banking, Credit Cards, Financial Services, FinTech', 'Apps, Internet, Social Media, Software',
    'Artificial Intelligence, Computer, Sales Automation, Software'
]

# Tried Electra... embeddings are not comparable for distance
electra_pre = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
electra_model = hub.load('https://tfhub.dev/google/electra_large/2')


def text_to_embeds_electra(text_list):
    text_embeddings = electra_model(electra_pre(text_list), training=False)['pooled_output']
    correlation = tf.convert_to_tensor([tf.keras.losses.cosine_similarity(-x, text_embeddings, axis=-1) for x in text_embeddings]).numpy()
    return 'electra', text_embeddings.numpy(), correlation


# emb_industry_e, corr_industry_e = text_to_embeds_electra(text_to_relate)


# Tried Albert... embeddings are not comparable for distance
def get_albert():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_xxlarge/3", trainable=False)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]  # [batch_size, 4096].
    # sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 4096].
    embedding_model = tf.keras.Model(text_input, pooled_output)
    return embedding_model


def text_to_embeds_albert(model, text_list):
    sentences = tf.constant(text_list)
    text_embeddings = model(sentences, training=False)
    correlation = tf.convert_to_tensor([tf.keras.losses.cosine_similarity(-x, text_embeddings, axis=-1) for x in text_embeddings]).numpy()
    return 'albert', text_embeddings.numpy(), correlation

# albert_model = get_albert()
# emb_ind_albert, corr_ind_albert = text_to_embeds_albert(albert_model, text_to_relate)


# Use Heatmapz for a better representation of heatmap
# def plot_corr_matrix_heatmapz(labels, corr_matrix, rotation, title):
#     plt.figure(figsize=(8, 8))
#
#     lx = []
#     ly = []
#     lc = []
#     for j in range(corr_matrix.shape[1]):
#         for i in range(corr_matrix.shape[0]):
#             lx.append(i)
#             ly.append(j)
#             lc.append(corr_matrix[i][j])
#
#     hmz.heatmap(lx, ly,
#                 color=lc, color_range=[0, 1],
#                 palette=sns.color_palette("YlOrRd", 256),
#                 # palette=sns.diverging_palette(20, 220, n=256),
#                 size=lc, size_range=[0, 1], size_scale=64
#                 # marker=marker,
#                 # x_order=data.columns,
#                 # y_order=data.columns[::-1],
#                 )
