import streamlit as st
import re
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances
#from IPython.display import clear_output
#import time

#################　関数　#################
def extract_matching_indices(df, columns, strings_in):

    matching_indices = set()
    
    # 指定されたカラムに対して処理を行う
    cnt = 0
    for col in columns:
        strings = strings_in[cnt]
        cnt = cnt + 1
        if col in df.columns:
            # 各セルの要素がnumpy.ndarrayの場合、文字列に変換してからチェックする
            for idx, value in df[col].iteritems():
                # ndarrayを文字列に変換して検索
                #print(col,type(value))
                if isinstance(value, np.ndarray):
                    value_str = np.array2string(value)
                    #print("aaa",value,value_str)
                else:
                    value_str = value
                # 特定の文字列が含まれているかチェック
                if any(s in value_str for s in strings):
                    matching_indices.add(idx)
    
    return list(matching_indices)


def PCA_2D_plot(dict_plot,data):
    # PCAで2次元に圧縮
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    df_reduced = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])

 
    # 全てのデータをグレーでプロット
    #ax.figure(figsize=(18, 16))

    #　カラーマップを分散
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(dict_plot))]

    # scatterプロットの作成
    fig, ax = plt.subplots()
    # 全てのデータをグレーでプロット
    ax.scatter(df_reduced['PCA1'], df_reduced['PCA2'], color='gray', label='Other Data', alpha=0.6)
    # カテゴリごとに色を変えてプロット
    cnt = 0
    for key, value in dict_plot.items():
        # 指定したインデックスのデータを赤で強調表示
        ax.scatter(df_reduced.loc[value, 'PCA1'], df_reduced.loc[value, 'PCA2'], 
                    color=colors[cnt], label=key, s=100)
        print(key)
        cnt = cnt + 1
    ax.set_title("PCA 2D Projection of High-Dimensional Data")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    st.pyplot(fig)

def make_clickable(url):
    return f'<a href="{url}" target="_blank">Link</a>'

# 重心計算の関数
def calculate_centroid(vectors):
    return np.mean(vectors, axis=0)

# 分類ごとに重心を計算し、しきい値処理を行う関数
def filter_by_cosine_threshold(A, threshold):
    grouped = A.groupby('classification')
    updated_embeddings = []
    
    for classification, group in grouped:
        vectors = np.array(group['embedding_v1'].tolist())
        publication_numbers = group['publication_number'].tolist()
        
        # 初回の重心を計算
        centroid = calculate_centroid(vectors)
        
        while True:
            # 重心からのコサイン距離を計算
            distances = cosine_distances([centroid], vectors)[0]
            
            # しきい値以内のベクトルのみを抽出
            within_threshold = distances <= threshold
            filtered_vectors = vectors[within_threshold]
            filtered_publication_numbers = [pub for pub, include in zip(publication_numbers, within_threshold) if include]
            
            # 重心を再計算
            new_centroid = calculate_centroid(filtered_vectors)
            
            # 重心が更新されない、またはすべてのベクトルがしきい値以内に収まる場合は終了
            if np.array_equal(new_centroid, centroid) or len(filtered_vectors) == len(vectors):
                centroid = new_centroid
                break
            
            # 重心の更新と、ベクトル集合の更新
            centroid = new_centroid
            vectors = filtered_vectors
            publication_numbers = filtered_publication_numbers
        
        # 重心に最もコサイン距離が近い行のpublication_numberを取得
        distances = cosine_distances([centroid], vectors)[0]
        closest_index = np.argmin(distances)
        closest_publication_number = publication_numbers[closest_index]
        
        # 最終的な重心をその分類のベクトルデータとして保存
        updated_embeddings.append({'classification': classification, 'embedding_v1': centroid, 'publication_number': closest_publication_number})
    
    # DataFrameとして結果を返す
    return pd.DataFrame(updated_embeddings)

# 分類が付与された最寄りの特許を検索し、コサイン類似度がthreshold以上ならその分類を採用
def classification_cosine_similarity(A,B,threshold):
    # 結果を保持するリスト
    most_similar_pub_nums = []
    most_similar_categories = []
    most_similarities = []
    A_ORG = A
    A = filter_by_cosine_threshold(A,threshold)
    

    # B の各行についてコサイン距離を計算して、最も類似するものを見つける
    for index, row in B.iterrows():
        max_similarity      = -1  # 最も高いコサイン類似度を格納する変数
        best_match_pub_num  = None
        best_match_category = None

        # B の publication_number に対応する A のベクトルを取得
        if (A_ORG['publication_number'] == row['publication_number']).any():
            b_vector = A_ORG.loc[A_ORG['publication_number'] == row['publication_number'], 'embedding_v1'].values[0]
            print(index+1,"/",len(B))

            # A の各ベクトルデータとのコサイン距離を計算
            for _, a_row in A.iterrows():
                if row['publication_number'] == a_row['publication_number'] or pd.isna(a_row['classification']):
                    continue  # Bのpublication_numberがAと同じ場合、またはカテゴリがNaNの場合はスキップ            
                similarity = 1 - cosine(b_vector, a_row['embedding_v1'])

                if similarity >= threshold and similarity > max_similarity:
                    max_similarity = similarity
                    best_match_pub_num = a_row['publication_number']
                    best_match_category = a_row['classification']

            most_similar_pub_nums.append(best_match_pub_num)
            most_similar_categories.append(best_match_category)
            most_similarities.append(max_similarity if max_similarity != -1 else np.nan)
            #clear_output(wait=True)
        else:
            most_similar_pub_nums.append('DBに案件が存在しません')
            most_similar_categories.append('DBに案件が存在しません')
            most_similarities.append(0)
        
    return most_similar_pub_nums,most_similar_categories,most_similarities
###############　session_state　###############
if 'open_state' not in st.session_state:
    st.session_state.open_state = False

####################  DB  #####################
##set_pkl_file_1 = "Pana_Denso_出願過去20年.pkl"

# 分割されたファイルのリスト
N=5
file_list = [f'chunk_{i}.pkl' for i in range(N)]

# 各ファイルを読み込んでリストに追加
loaded_chunks = []
for file_name in file_list:
    with open(file_name, 'rb') as f:
        loaded_chunks.append(pickle.load(f))

# チャンクを結合して元のDataFrameを再構築
all_df_tmp1 = pd.concat(loaded_chunks, ignore_index=True)

# 結果のDataFrameを表示
#print(len(all_df_tmp1))

#################　pickle読込み ################
#all_df_tmp1   = pd.read_pickle(set_pkl_file_1)
all_vector_df = all_df_tmp1['embedding_v1'].tolist()
vectors       = np.array(all_df_tmp1['embedding_v1'].tolist())

#################　Main　#################
if st.session_state.open_state == False:

    search_column = 'publication_number'
    search_string = 'JP-2019056840-A'
    N_default     = 30 # TOPいくつ選ぶか
    threshold     = 0.99999 #コサイン類似度が最も高い（一定値以上）インデックスを取得

    st.write("**～～～対象データ～～～**")
    st.write("**日本／パナソニックorデンソーのみ／過去20年**")
    #st.write("Streamlit version:", st.__version__)
    #st.write(time.__version__)
    st.write("**（機能１）公開番号を指定して類似特許を検出します**")
    st.write("データベースの読み込み件数",len(all_df_tmp1))
    st.write("\n\n")

    search_string = st.text_input("公開番号を1件入力してください　例：JP-2019056840-A",value=search_string)
    N = st.number_input('類似の特許を検出する件数を入力してください（100件まで）', min_value=0, max_value=100, value=N_default)
    load_flg1 = st.button('検索開始', key="load_flg1")
    load_flg2 = st.button('クリア', key="load_flg2")

    if load_flg2:
        load_flg1=False

    if load_flg1:


        #　以下処理をするDF
        df = all_df_tmp1

        # 検索条件に一致する行のインデックスを抽出
        matching_indices = df[df[search_column].str.contains(search_string, na=False)].index

        flg1 = 1
        if matching_indices.empty:
            st.write("データベースに特許がみつかりません")
            st.write("（存在する公開番号を入れてください　例：JP-2019056840-A）")
            flg1 = 0
        else:
            # インデックスが1つだけなら、それをintに変換
            if len(matching_indices) == 1:
                single_index = int(matching_indices[0])
            else:
                pass
            #st.write(single_index) 

        if flg1==1:
            # params
            target_vector_index = single_index

            # ある行のベクトル（例: インデックス0のベクトル）
            target_vector = vectors[target_vector_index].reshape(1, -1)

            # 全てのベクトルとターゲットベクトルのコサイン類似度を計算
            cos_similarities = cosine_similarity(target_vector, vectors).flatten()

            # ターゲット行自身を除くためにインデックスを無効化
            cos_similarities[target_vector_index] = -1

            # コサイン類似度が最も高い（＝距離が最も近い）インデックスを取得
            most_similar_index = np.argmax(cos_similarities)

            # 最もコサイン距離が近い行を抽出
            most_similar_row = df.iloc[most_similar_index]

            # コサイン類似度が高い順にN個のインデックスを取得（例：N=3）
            nearest_indices = np.argsort(-cos_similarities)[:N]

            # コサイン類似度が最も高い（一定値以上）インデックスを取得
            # nearest_indices = [i for i, value in enumerate(cos_similarities) if value >= threshold]

            # ターゲット
            #print("SSSS",df.iloc[target_vector_index])
            st.write("検索対象のリンク\n",df.loc[target_vector_index,'url'])
            #st.write("データベースに含まれるassignee",df.loc[target_vector_index,'assignee'],"\n\n")

            # 最もコサイン類似度が高いN個の行を抽出
            nearest_rows = df.iloc[nearest_indices]
            #st.write("cos_similarities\n",cos_similarities[nearest_indices],len(cos_similarities[nearest_indices]))
            pd.set_option('display.max_rows', None)
            tmp_list = cos_similarities[nearest_indices]

            # 表示
            filtered_rows = df.loc[nearest_indices, ['url','publication_number','assignee']]
            filtered_rows = filtered_rows.reset_index(drop=True)
            filtered_rows['cos_similarities'] = tmp_list

            # URL列をクリック可能なリンクに変換
            filtered_rows['url'] = filtered_rows['url'].apply(make_clickable)
            st.write(f"コサイン類似度が高い特許を　{len(filtered_rows)}　件、抽出しました")
            st.write(filtered_rows.to_html(escape=False, index=False), unsafe_allow_html=True)

###############################################
st.write("\n\n")
st.write("**（機能２）分類ファイルを読み込んで、テストファイルを分類します**")
st.write("\n\n")

# publication_number と　classification の対応をCSVで読み込む（ANSI）

###### CSVデフォルト
classification_file_name = "classification.csv"
test_file_name = "test_data.csv"
#classification_file_name = st.text_input("公開番号と分類を記載したCSVを指定してください",value=classification_file_name)
#test_file_name = st.text_input("分類対象の公開番号を記載したCSVを指定してください",value=test_file_name)
df_classification = pd.read_csv(classification_file_name)
df_test_data = pd.read_csv(test_file_name)

###### CSVアップロード
uploaded_file1 = st.file_uploader(f"公開番号と分類を記載したCSVを指定してください ({classification_file_name})", type="csv", key="uploaded_file1")
if uploaded_file1 is not None:
    # CSV ファイルを pandas DataFrame に読み込む
    df_classification = pd.read_csv(uploaded_file1)

uploaded_file2 = st.file_uploader(f"分類対象の公開番号を記載したCSVを指定してください ({test_file_name})", type="csv", key="uploaded_file2")
if uploaded_file2 is not None:
    # CSV ファイルを pandas DataFrame に読み込む
    df_test_data = pd.read_csv(uploaded_file2)

load_flg3 = st.button('分類開始', key="load_flg3")

if load_flg3:
    # 重複行を検出
    #duplicate_rows = df_classification[df_classification.duplicated(subset=['publication_number', 'classification'], keep=False)]
    #print("重複行があります\n",duplicate_rows)

    # 重複行を削除
    df_unique = df_classification.drop_duplicates(subset=['publication_number', 'classification'], keep='first')

    # publication_numbergが重複していたら、最初に出てきた行を残す
    df_unique = df_unique.drop_duplicates(subset=['publication_number'], keep='first')
    df_unique = df_unique.reset_index(drop=True) #インデックスをリセット
    #print(df_unique.head(10))

    # publication_numberをキーにして、dataframeを結合
    df_merged = pd.merge(all_df_tmp1, df_unique, on='publication_number', how='left')
    #print(df_merged.head(5))

    # classificationgがユニークな列のインデックスを抽出
    unique_index_dict = {value: df_merged.index[df_merged['classification'] == value].tolist() for value in df_merged['classification'].unique() if pd.notna(value)}
    print(unique_index_dict)

    # PCA_2D_plot
    st.write("対象データ全体と分類のマップ")
    dict_plot = unique_index_dict
    PCA_2D_plot(dict_plot,vectors)

    ############## 分類処理　##############
    A = df_merged
    B = df_test_data
    threshold = 0.8 # 所定の閾値を設定

    most_similar_pub_nums,most_similar_categories,most_similarities = classification_cosine_similarity(A,B,threshold)

    # B に新しい列を追加
    print("END!")
    B['most_similar_publication_number'] = most_similar_pub_nums
    B['classification'] = most_similar_categories
    B['cosine_similarity'] = most_similarities

    csv_text = B.to_csv(index=False)
    st.text_area("結果をCSVでコピーしてください", csv_text, height=200)