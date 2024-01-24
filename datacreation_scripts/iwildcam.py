import os

import pandas as pd
import argparse
import pdb
import pickle
from sklearn.model_selection import train_test_split


def main(args):
    iwildcam_template = [
        lambda c: f"a photo of {c}.", lambda c: f"{c} in the wild."
    ]
    if args.mode == 'train':
        label_to_name = pd.read_csv(args.label_file_ori)
    else:
        label_to_name = pd.read_csv(args.label_file_ori.replace('labels.csv', 'labels_new.csv'))

    label_to_name = label_to_name[label_to_name['y'] < 99999]
    label_to_name['prompt1'] = label_to_name['english'].map(
        iwildcam_template[0])
    label_to_name['prompt2'] = label_to_name['english'].map(
        iwildcam_template[1])

    sp_name = set(label_to_name['name'].values.tolist())

    img_sp_folder = os.listdir(args.input_folder)
    img_sp_folder = [item for item in img_sp_folder if item.replace('_', ' ') in sp_name]

    list_result = []
    list_y = []
    if args.mode == 'train':
        if args.curriculum:
        # if True:
            clip_path = '../data/metadata/clip_score_train.pkl'
            threshold = 0.25
            dict_filt = dict()
            if os.path.exists(clip_path):
                with open(clip_path, 'rb') as f:
                    dict_clip_res = pickle.load(f)
                list_filtered = list(dict_clip_res.items())
                list_filtered = [[item[0].split('='), item[1][0][0]] for item in list_filtered]  # list_filtered = [[sp_name, cate, img_id], score]
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

            for cur_sp_f in img_sp_folder:
                cur_sp_path = os.path.join(args.input_folder, cur_sp_f)
                cur_sp_name = cur_sp_f.replace('_', ' ')
                cur_y = label_to_name[label_to_name['name']==cur_sp_name]['y'].values[0]
                if cur_y not in list_y:
                    list_y.append(cur_y)
                list_img_cate = os.listdir(cur_sp_path)
                for cate in list_img_cate:
                    cur_strength = int(cate.split('_')[0].replace('Strength', ''))
                    cur_cate_path = os.path.join(cur_sp_path, cate)
                    list_sub_img = os.listdir(cur_cate_path)
                    list_sub_img = [item for item in list_sub_img if 'jpg' in item]

                    for img_name in list_sub_img:
                        cur_img_path = os.path.join(cur_cate_path, img_name)
                        if len(dict_filt) > 0:
                            if cate in dict_filt:
                                if cur_sp_f in dict_filt[cate]:
                                    if img_name.replace('.jpg', '') in dict_filt[cate][cur_sp_f]:
                                        list_result.append([cur_y, cur_img_path, cur_strength])

                        else:
                            list_result.append([cur_y, cur_img_path, cur_strength])
        
        # #############################################
        # # if using part of the original training data
        # img_sp_folder_ori = os.listdir("../data/train")
        # img_sp_folder_ori = [item for item in img_sp_folder_ori if item in img_sp_folder]
        # cur_strength = 0
        # for cur_sp_f in img_sp_folder_ori:
        #     cur_sp_path = os.path.join("../data/train", cur_sp_f)
        #     cur_sp_name = cur_sp_f.replace('_', ' ')
        #     cur_y = label_to_name[label_to_name['name']==cur_sp_name]['y'].values[0]
        #     if cur_y not in list_y:
        #         list_y.append(cur_y)
        #     list_imgs = os.listdir(cur_sp_path)
        #     list_imgs = [item for item in list_imgs if 'jpg' in item]
        #     for img_name in list_imgs:
        #         cur_img_path = os.path.join(cur_sp_path, img_name)
        #         list_result.append([cur_y, cur_img_path, cur_strength])

        # #############################################
        # # if using all training data
        df_train_ori = pd.read_csv('./datasets/csv/iwildcam_v2.0/train.csv', sep='\t')
        del df_train_ori['title']
        df_train_ori.drop_duplicates(subset=['filepath', 'label'], keep='last', inplace=True)
        df_train_ori.rename({'filepath': 'filename','label': 'y'}, axis='columns', inplace=True)
        df_train_ori['strength'] = 0
        df_train_ori = df_train_ori[['y', 'filename', 'strength']]
        cur_train_ori = df_train_ori.values.tolist()
        list_result.extend(cur_train_ori)

    else:
        img_sp_folder_train = os.listdir("../data/train_new")
        img_sp_folder_train = [item for item in img_sp_folder if item.replace('_', ' ') in sp_name]

        img_sp_folder_test = os.listdir(args.input_folder)
        img_sp_folder_test = [item for item in img_sp_folder_test if item in img_sp_folder_train]
        
        cur_strength = 0
        for cur_sp_f in img_sp_folder_test:
            cur_sp_path = os.path.join(args.input_folder, cur_sp_f)
            cur_sp_name = cur_sp_f.replace('_', ' ')
            cur_y = label_to_name[label_to_name['name']==cur_sp_name]['y'].values[0]
            list_imgs = os.listdir(cur_sp_path)
            list_imgs = [item for item in list_imgs if 'jpg' in item]
            for img_name in list_imgs:
                cur_img_path = os.path.join(cur_sp_path, img_name)
                list_result.append([cur_y, cur_img_path, cur_strength])

    df = pd.DataFrame(list_result, columns=['y', 'filename', 'strength'])
    
    if args.random:
        print(len(df))
        df_group = df.groupby('strength')

        # Create a list of dataframes
        df_list = pd.DataFrame()

        # Iterate over the groups
        for name, group in df_group:
            print(f"len group {len(group)}")
            # Split the group into two dataframes based on a random split
            train = group.sample(frac=0.15, random_state=1)
            print(f"len train {len(train)}")

            # Add the dataframes to the list
            # df_list.append(df_train)
            df_list = pd.concat([df_list, train])

        # Create a new dataframe from the list
        df = df_list
        print(len(df))

    if args.mode == 'train':
        label_to_name = label_to_name[label_to_name['y'].isin(list_y)]
        # pdb.set_trace()
        new_path = args.label_file_ori.replace('labels.csv', 'labels_new.csv')
        label_to_name.to_csv(new_path, index=False)
    # assert len(df) == 129809, 'number of samples incorrect'

    df1 = pd.merge(df, label_to_name[['y', 'prompt1']],
                   on='y').rename({'prompt1': 'title'}, axis='columns')
    df2 = pd.merge(df, label_to_name[['y', 'prompt2']],
                   on='y').rename({'prompt2': 'title'}, axis='columns')

    # assert len(df1) == 129809, 'number of samples incorrect'
    # assert len(df2) == 129809, 'number of samples incorrect'

    df_final = pd.concat((df1, df2))[['filename', 'title', 'y', 'strength']]

    del df1
    del df2
    del df

    df_final = df_final.rename({'filename': 'filepath','y': 'label'},
                               axis='columns')[['title', 'filepath', 'label', 'strength']]

    # assert len(df_final) == 129809 * 2, 'number of samples incorrect'

    if args.mode == "train":
        df_final.to_csv(os.path.join(args.save_folder, 'train.csv'), sep='\t', index=False, header=True)
    else:
        df_final.to_csv(os.path.join(args.save_folder, 'test.csv'), sep='\t', index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--mode',
                        default='train')    
    parser.add_argument('--curriculum', action=argparse.BooleanOptionalAction)
    parser.add_argument('--random', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_folder',
                        default='./datasets/csv/iwildcam_v2.0/')
    parser.add_argument('--input_folder',
                        default='../data/train_new')
    parser.add_argument('--label_file_ori',
                        default='./src/datasets/iwildcam_metadata/labels.csv')
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)

    main(args)
