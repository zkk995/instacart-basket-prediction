import os
import gc
import numpy as np
import pandas as pd


product_df = pd.read_csv('../../data/processed/product_data.csv', usecols=['user_id', 'product_id', 'label'])
products = pd.read_csv('../../data/raw/products.csv')
product_df = product_df.merge(products, how='left', on='product_id')

orders = pd.read_csv('../../data/raw/orders.csv')
orders = orders[orders['eval_set'].isin({'train', 'test'})]
product_df = product_df.merge(orders[['user_id', 'order_id']], how='left', on='user_id').reset_index(drop=True)
product_df['is_none'] = (product_df['product_id'] == 0).astype(int)


# nn feature representations
sgns_matrix = np.load('../sgns/predictions/product_embeddings.npy')
product_emb_df = pd.DataFrame(sgns_matrix, columns=['sgns_{}'.format(i) for i in range(sgns_matrix.shape[1])])
product_emb_df['product_id'] = np.arange(sgns_matrix.shape[0])
product_df = product_df.merge(product_emb_df, how='left', on='product_id')

nnmf_p_matrix = np.load('../nnmf/predictions/product_embeddings.npy')
product_emb_df = pd.DataFrame(nnmf_p_matrix, columns=['nnmf_product_{}'.format(i) for i in range(nnmf_p_matrix.shape[1])])
product_emb_df['product_id'] = np.arange(nnmf_p_matrix.shape[0])
product_df = product_df.merge(product_emb_df, how='left', on='product_id')

nnmf_u_matrix = np.load('../nnmf/predictions/user_embeddings.npy')
user_emb_df = pd.DataFrame(nnmf_u_matrix, columns=['nnmf_user_{}'.format(i) for i in range(nnmf_u_matrix.shape[1])])
user_emb_df['user_id'] = np.arange(nnmf_u_matrix.shape[0])
product_df = product_df.merge(user_emb_df, how='left', on='user_id')

del sgns_matrix, product_emb_df, nnmf_u_matrix, user_emb_df, orders
prefix = 'rnn_product'
print(prefix)
h_df = pd.DataFrame(np.load('../rnn_product/predictions/final_states.npy')).add_prefix('{}_h'.format(prefix))
h_df['user_id'] = np.load('../rnn_product/predictions/user_ids.npy')
h_df['product_id'] = np.load('../rnn_product/predictions/product_ids.npy')
product_df = product_df.merge(h_df, how='left', on=['user_id', 'product_id'])

prefix = 'rnn_product_bmm'
print(prefix)
h_df = pd.DataFrame(np.load('../rnn_product/predictions_bmm/final_states.npy')).add_prefix('{}_h'.format(prefix))
h_df['user_id'] = np.load('../rnn_product/predictions_bmm/user_ids.npy')
h_df['product_id'] = np.load('../rnn_product/predictions_bmm/product_ids.npy')
#h_df['{}_prediction'.format(prefix)] = np.load('../rnn_product/predictions_bmm/predictions.npy')
product_df = product_df.merge(h_df, how='left', on=['user_id', 'product_id'])

prefix = 'rnn_aisle'
print(prefix)
h_df = pd.DataFrame(np.load('../rnn_aisle/predictions/final_states.npy')).add_prefix('{}_h'.format(prefix))
h_df['user_id'] = np.load('../rnn_aisle/predictions/user_ids.npy')
h_df['aisle_id'] = np.load('../rnn_aisle/predictions/aisle_ids.npy')
h_df['{}_prediction'.format(prefix)] = np.load('../rnn_aisle/predictions/predictions.npy')
product_df = product_df.merge(h_df, how='left', on=['user_id', 'aisle_id']).fillna(-1)

prefix = 'rnn_department'
print(prefix)
h_df = pd.DataFrame(np.load('../rnn_department/predictions/final_states.npy')).add_prefix('{}_h'.format(prefix))
h_df['user_id'] = np.load('../rnn_department/predictions/user_ids.npy')
h_df['department_id'] = np.load('../rnn_department/predictions/department_ids.npy')
h_df['{}_prediction'.format(prefix)] = np.load('../rnn_department/predictions/predictions.npy')
product_df = product_df.merge(h_df, how='left', on=['user_id', 'department_id']).fillna(-1)

prefix = 'rnn_order_size'
print(prefix)
h_df = pd.DataFrame(np.load('../rnn_order/predictions/final_states.npy')).add_prefix('{}_h'.format(prefix))
h_df['user_id'] = np.load('../rnn_order/predictions/user_ids.npy')
h_df['{}_prediction'.format(prefix)] = np.load('../rnn_order/predictions/predictions.npy')
product_df = product_df.merge(h_df, how='left', on='user_id')

product_df.to_pickle('product_df.pkl')
prefix = 'rnn_order_size_gmm'
print(prefix)
h_df = pd.DataFrame(np.load('../rnn_order/predictions_gmm/final_states.npy')).add_prefix('{}_h'.format(prefix))
h_df['user_id'] = np.load('../rnn_order/predictions_gmm/user_ids.npy')
product_df = product_df.merge(h_df, how='left', on='user_id')
del h_df
gc.collect()
print('save product_df')

drop_cols = [
    'label',
    'user_id',
    'product_id',
    'order_id',
    'product_name',
    'aisle_id',
    'department_id',
]
user_id = product_df['user_id']
product_id = product_df['product_id']
order_id = product_df['order_id']
label = product_df['label']

product_df.drop(drop_cols, axis=1, inplace=True)
features = product_df.values
feature_names = product_df.columns.values
feature_maxs = features.max(axis=0)
feature_mins = features.min(axis=0)
feature_means = features.mean(axis=0)

if not os.path.isdir('data'):
    os.makedirs('data')

np.save('data/user_id.npy', user_id)
np.save('data/product_id.npy', product_id)
np.save('data/order_id.npy', order_id)
np.save('data/features.npy', features)
np.save('data/feature_names.npy', product_df.columns)
np.save('data/feature_maxs.npy', feature_maxs)
np.save('data/feature_mins.npy', feature_mins)
np.save('data/feature_means.npy', feature_means)
np.save('data/label.npy', label)
