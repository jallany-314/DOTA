import csv
import random

machine_file_path = r'D:\学习\毕业论文\数据集\ali cluster\machine_meta.csv'
task_file_path = r'D:\学习\毕业论文\数据集\ali cluster\batch_task.csv'

mf_pre_head = r'D:\学习\毕业论文\数据集\ali cluster\machine_meta_'
tf_pre_head = r'D:\学习\毕业论文\数据集\ali cluster\batch_task_'
pre_file_tail = '.csv'

machine_file = open(machine_file_path, 'r')
task_file = open(task_file_path, 'r')

# read total message from original file
print('start read machine file...')
mf_csv = list(csv.reader(machine_file))
print('start read task file...')
tf_cvs = list(csv.reader(task_file))

# delete space in message
tf_cvs = [elem for elem in tf_cvs if elem.count('') == 0]

# set chosen counts
mf_counts = [100, 200, 500, 1000, 2000, 5000, 10000]
tf_counts = [100, 200, 500, 1000, 2000, 5000, 10000]

# choose message from original file and generate new files
for n in mf_counts:
    mf_pre = open(mf_pre_head + str(n) + pre_file_tail, 'w', newline='')
    contents_choose = random.sample(mf_csv, n)
    contents_choose1 = [elem[4:6] for elem in contents_choose]
    writer = csv.writer(mf_pre)
    writer.writerows(contents_choose1)
    print('total:', n, 'machine finish!')
    mf_pre.close()

for n in tf_counts:
    tf_pre = open(tf_pre_head + str(n) + pre_file_tail, 'w', newline='')
    contents_choose = random.sample(tf_cvs, n)
    contents_choose1 = [[elem[5]] + elem[7:9] for elem in contents_choose]
    writer = csv.writer(tf_pre)
    writer.writerows(contents_choose1)
    print('total:', n, 'task finish!')
    tf_pre.close()

machine_file.close()
task_file.close()
