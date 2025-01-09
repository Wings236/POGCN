import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle

root_path = './'
data_name = 'tenrec' # dataset name
data_path = os.path.join(root_path, data_name)
if not os.path.exists(data_path):
    os.mkdir(data_path)

df = pd.read_csv(data_name+'.csv')
behaviors = df.columns[2:].to_list()
print(f"{data_name} behaviors:", behaviors)
behaviors_path = [os.path.join(data_path, b) for b in behaviors]
for b_path in behaviors_path:
    if not os.path.exists(b_path):
        os.mkdir(b_path)

group = df.groupby('user_id')
least_click_num = 20
test_size = 0.2
test_valid_size, test_test_size = 0.5, 0.5

# Data preprocessing
userid_map, itemid_map = dict(), dict()
new_userid, new_itemid = 0, 0
all_data = []
print(f"the least number of interactions: {least_click_num}")
for user_id, temp in tqdm(group, ncols=80):
    pd_true_filter = ''
    for b in behaviors:
        pd_true_filter += f"(temp['{b}']==True) |"
    pd_true_filter = eval(pd_true_filter[:-1])
    new_temp = temp[pd_true_filter]   
    interaction_num = new_temp.shape[0]
    if interaction_num>=least_click_num:
        for line in new_temp.itertuples():
            user_id = line.user_id
            item_id = line.item_id
            if userid_map.get(user_id) is None:
                userid_map[user_id] = new_userid
                new_userid += 1
            if itemid_map.get(item_id) is None:
                itemid_map[item_id] = new_itemid
                new_itemid += 1
            behaviors_flag = [getattr(line, b) for b in behaviors]
            temp_list = [userid_map[user_id], itemid_map[item_id]]
            temp_list.extend(behaviors_flag)
            all_data.append(temp_list)
m_user = max(userid_map.values()) + 1
n_item = max(itemid_map.values()) + 1
print(f'total num:{len(all_data)}, user:{m_user}, item:{n_item}, sparsity:{len(all_data)/m_user/n_item}')
pickle.dump({"user_num":m_user, "item_num":n_item}, open(os.path.join(data_name,"ui_num.pkl"), "wb"))


# Spilt dataset
print(f"Dataset spilt, test set ratio is {test_size:.2f}")
all_data_np = np.array(all_data)
length = len(all_data)
test_idx = sorted(np.random.choice(length, int(length*test_size), replace=False))
train_idx = sorted(list(set(list(range(length))) - set(test_idx)))
train_data = all_data_np[train_idx]
test_data = all_data_np[test_idx]
print(f"train dataset num:{len(train_data)}, test dataset num:{len(test_data)}")
# valid and test dataset
test_length = len(test_data)
test_valid_idx = sorted(np.random.choice(test_length, int(test_length*test_valid_size), replace=False))
test_test_idx = sorted(list(set(list(range(test_length))) - set(test_valid_idx)))
test_valid_data = test_data[test_valid_idx]
test_test_data = test_data[test_test_idx]
print(f"valid datast num:{len(test_valid_data)}, test dataset num:{len(test_test_data)}")

behaviors = df.columns[2:].to_list()
behavior_num = [0 for _ in range(len(behaviors))]

# train valid test
data_list = [train_data, test_valid_data, test_test_data]
data_type = ['train', 'valid', 'test']
for data, d_type in zip(data_list, data_type):
    print(f"{d_type} dataset output")
    data_length = len(data)
    output_file = [open(f'{b_path}/{d_type}.txt', 'w') for b_path in behaviors_path]
    for idx, data_line in tqdm(enumerate(data), total=data_length, ncols=80):
        user_id = int(data_line[0])
        item_id = int(data_line[1])
        flag_list = [int(data_line[fidx+2]) for fidx in range(len(behaviors))]
        # A line start
        if idx == 0 or data[idx-1][0] != user_id :
            flag_line_list = [str(user_id) for _ in range(len(behavior_num))]
        
        # items 
        for fidx, temp_flag in enumerate(flag_list):
            if temp_flag == 1:
                flag_line_list[fidx] += ' ' + str(item_id)
                behavior_num[fidx] += 1

        if idx == data_length-1 or data[idx+1][0] != user_id:
            for temp_line, temp_file in zip(flag_line_list, output_file):
                temp_line += '\n'
                temp_file.write(temp_line)
    # file close
    for temp_file in output_file:
        temp_file.close()

# infomation of total behaviros output
output_line = [f"{b} total num:{behavior_num[idx]}" for idx, b in enumerate(behaviors)]
for o_line in output_line:
    print(o_line)