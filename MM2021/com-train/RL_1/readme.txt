train.py转变为solution.py需要修改如下：

1. 删除main.py函数
2. 删除所有的print()
3. 修改模型NN_MOLDE

SUMMARY_DIR = './model/v2'
RESULT_DIR = './model/v2.csv'
CC_ADD_RATE = 100
CC_MINUS_RATE = 100 

TO

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
NN_MODEL = current_dir+"/model/v1/nn_model_ep_3.ckpt"
