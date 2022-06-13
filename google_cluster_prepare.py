import json
import random

ori_file_path = r'D:\学习\毕业论文\数据集\google cluster\instance_events-000000000000.json'

ori_file = open(ori_file_path, 'r')
pre_file_head = r'D:\学习\毕业论文\数据集\google cluster\instance_events-change'
pre_file_tail = '.json'

# read total message from original file
contents = ori_file.readlines()

# set chosen counts
choose_counts = [100, 200, 500, 1000, 2000, 5000, 10000]

# choose message from original file and generate new files
for n in choose_counts:
    pre_file = open(pre_file_head + str(n) + pre_file_tail, 'w')
    contents_choose = random.sample(contents, n)
    contents_new = []
    count = 0
    for content in contents_choose:
        content_dict = json.loads(content)
        content_dict1 = {k: v for k, v in content_dict.items() if k == 'time' or k == 'resource_request'}
        content1 = json.dumps(content_dict1) + '\n'
        contents_new.append(content1)
        count += 1
        print('total:', n, 'finish:', count)
    pre_file.writelines(contents_new)
    pre_file.flush()
    pre_file.close()

# close original file
ori_file.close()
