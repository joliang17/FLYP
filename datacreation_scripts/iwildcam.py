import os

import pandas as pd
import argparse
import pdb
import pickle
from sklearn.model_selection import train_test_split
from typing import List, Dict
from tqdm import tqdm


def filter_img(clip_path: str, threshold: float):
    dict_filt = dict()
    img_cnt = 0
    if os.path.exists(clip_path):
        with open(clip_path, 'rb') as f:
            dict_clip_res = pickle.load(f)
        list_filtered = list(dict_clip_res.items())
        list_filtered = [[item[0].split('='), item[1][0][0]] for item in
                         list_filtered]  # list_filtered = [[sp_name, cate, img_id], score]
        list_filtered = [item[0] for item in list_filtered if item[1] >= threshold]
        for pair in list_filtered:
            cur_sp = pair[0]
            cur_cate = pair[1]
            cur_imgid = pair[2]
            if cur_cate not in dict_filt:
                dict_filt[cur_cate] = dict()
            if cur_sp not in dict_filt[cur_cate]:
                dict_filt[cur_cate][cur_sp] = []
            dict_filt[cur_cate][cur_sp].append(cur_imgid)
            img_cnt += 1
    return dict_filt, img_cnt


def filter_generated_img(pkl_path: str, dict_filt: float):
    img_cnt = 0
    dict_filt_new = dict()
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            set_img_id = pickle.load(f)

        dict_img_id = dict()
        for cur_cate, dict_sub in dict_filt.items():
            dict_filt_new[cur_cate] = dict()
            for cur_sp, list_img in dict_sub.items():
                dict_filt_new[cur_cate][cur_sp] = []
                for img_id in list_img:
                    if img_id in set_img_id:
                        dict_filt_new[cur_cate][cur_sp].append(img_id)
                        img_cnt += 1
    return dict_filt_new, img_cnt, len(set_img_id)


def merge_with_prompt(df, label_to_name, merge_type='train'):
    # merge prompts
    df1 = pd.merge(df, label_to_name[['y', 'prompt1']], on='y').rename({'prompt1': 'title'}, axis='columns')
    df2 = pd.merge(df, label_to_name[['y', 'prompt2']], on='y').rename({'prompt2': 'title'}, axis='columns')
    df_final = pd.concat((df1, df2))[['filename', 'title', 'y', 'strength', 'guidance', 'img_id']]

    del df1
    del df2
    del df

    df_final = df_final.rename({'filename': 'filepath', 'y': 'label'}, axis='columns')[
        ['title', 'filepath', 'label', 'strength', 'guidance', 'img_id']]
    print(f"length of final {merge_type}.csv: {len(df_final)}")
    return df_final


def main(args):
    iwildcam_template = [lambda c: f"a photo of {c}.", lambda c: f"{c} in the wild."]
    # for training and curriculum progress evaluation
    label_to_name = pd.read_csv("./src/datasets/iwildcam_metadata/labels.csv")

    label_to_name = label_to_name[label_to_name['y'] < 99999]
    label_to_name['prompt1'] = label_to_name['english'].map(iwildcam_template[0])
    label_to_name['prompt2'] = label_to_name['english'].map(iwildcam_template[1])

    sp_name = set(label_to_name['name'].values.tolist())

    img_sp_folder = os.listdir(args.input_folder)
    img_sp_folder = [item for item in img_sp_folder if item.replace('_', ' ') in sp_name]

    list_result = []
    if args.curriculum:
        clip_path = f'{args.data_folder}/data/metadata/clip_score_train_new.pkl'
        threshold = 0.25
        Dict_filt, img_cnt = filter_img(clip_path, threshold)

        if args.gene_constr != '':
            Dict_filt, img_cnt_1, uniq_cnt = filter_generated_img(args.gene_constr, Dict_filt)

        all_cnt = 0
        filtered_cnt = 0
        for cur_sp_f in tqdm(img_sp_folder):
            cur_sp_path = os.path.join(args.input_folder, cur_sp_f)
            cur_sp_name = cur_sp_f.replace('_', ' ')
            cur_y = label_to_name[label_to_name['name'] == cur_sp_name]['y'].values[0]

            list_img_cate = os.listdir(cur_sp_path)
            for cate in list_img_cate:
                cur_strength = int(cate.split('_')[0].replace('Strength', ''))
                cur_cate_path = os.path.join(cur_sp_path, cate)
                list_sub_img = os.listdir(cur_cate_path)
                list_sub_img = [item for item in list_sub_img if 'jpg' in item]

                for img_name in list_sub_img:
                    cur_img_path = os.path.join(cur_cate_path, img_name)
                    img_name = img_name.replace('.jpg', '')
                    all_cnt += 1
                    if len(Dict_filt) > 0:
                        if cate in Dict_filt and cur_sp_f in Dict_filt[cate] and img_name.replace('.jpg', '') in \
                                Dict_filt[cate][cur_sp_f]:
                            list_result.append([cur_y, cur_img_path, cur_strength])
                            filtered_cnt += 1

                    else:
                        list_result.append([cur_y, cur_img_path, cur_strength])

    #############################################
    # using all training data
    print(f'generated data: {len(list_result)}')
    df_train_ori = pd.read_csv(f'{args.data_folder}/data/iwildcam/iwildcam_v2.0/train.csv', sep='\t')
    del df_train_ori['title']
    df_train_ori.drop_duplicates(subset=['filepath', 'label'], keep='last', inplace=True)
    df_train_ori.rename({'filepath': 'filename', 'label': 'y'}, axis='columns', inplace=True)
    df_train_ori['strength'] = 0
    df_train_ori = df_train_ori[['y', 'filename', 'strength']]
    cur_train_ori = df_train_ori.values.tolist()
    list_result.extend(cur_train_ori)
    print(f'Total data: {len(list_result)}')

    df = pd.DataFrame(list_result, columns=['y', 'filename', 'strength'])
    df.loc[:, 'guidance'] = df['strength'].apply(lambda x: 100 - int(x))
    df.loc[:, 'img_name'] = df['filename'].apply(lambda x: x.split('/')[-1].replace('.jpg', ''))

    print('adding img id')
    # change img_name to int img_id
    df_count = df.groupby(['img_name']).count()['guidance']
    list_guid_img_name = list(df_count[df_count > 1].index)  # largest 7715
    Dict_img_id = {list_guid_img_name[i]: i for i in range(len(list_guid_img_name))}

    list_ori_guid = list(df_count[df_count == 1].index)  # largest 124898  img id start from
    Dict_img_id_ori = {list_ori_guid[i]: i + 1 for i in range(len(list_ori_guid))}

    df.loc[:, 'img_id'] = df['img_name'].apply(lambda x: Dict_img_id[x] if x in Dict_img_id else -Dict_img_id_ori[x])

    # TODO: select image_id with all guidance
    print(f"selecting images with all guidance for guidance selection")
    df_count = df.groupby(['img_name', 'guidance']).count().reset_index()
    df_count = df_count.groupby(['img_name',]).count()['guidance'].reset_index()
    sel_img = df_count[df_count['guidance'] == 7].sample(n=1000, replace=False, ignore_index=True, random_state=42)['img_name'].values.tolist()
    df_sel = df[df['img_name'].isin(sel_img)].reset_index(drop=True)

    # merge prompts
    df_final = merge_with_prompt(df, label_to_name, merge_type='train')
    df_final.to_csv(os.path.join(args.save_folder, f'train.csv'), sep='\t', index=False, header=True)
    
    df_sel_final = merge_with_prompt(df_sel, label_to_name, merge_type='curriculum')
    df_sel_final.to_csv(os.path.join(args.save_folder, f'curriculum.csv'), sep='\t', index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--curriculum', action=argparse.BooleanOptionalAction)
    parser.add_argument('--gene_constr', default='')
    parser.add_argument('--total_train', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_folder', default='../data/metadata/clip_progress_difficult_v2')
    parser.add_argument('--input_folder', default='../data/train_new')
    parser.add_argument('--data_folder', default='/fs/nexus-scratch/yliang17/Research/diffusion/gene_diffcls')
    args = parser.parse_args()
    # args.gene_constr = '../data/metadata/used_imgid/used_imgid_v2.pkl'

    os.makedirs(args.save_folder, exist_ok=True)

    main(args)
