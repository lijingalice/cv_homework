# Q for teacher: in cal_step_gradient, it seems the for-loop is faster than reduce, do you know why?
import numpy as np
import time
from numpy import random
from functools import reduce
random.seed(123)

def inference(w, b, x):        # inference, test, predict, same thing. Run model after training
    pred_y = w * x + b
    return pred_y

def eval_loss(w, b, x_list, gt_y_list):
    return 0.5*np.mean(np.square(inference(w,b,x_list) - gt_y_list))

def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    batch_size = len(batch_x_list)
    pred_y = inference(w,b,batch_x_list)
    avg_dw = np.mean((pred_y - batch_gt_y_list) * batch_x_list)
    avg_db = np.mean((pred_y - batch_gt_y_list))
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b

def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    num_samples = len(x_list)
    t1=time.time()
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        w, b = cal_step_gradient(x_list[batch_idxs], gt_y_list[batch_idxs], w, b, lr)
        #print('w:{0}, b:{1}'.format(w, b))
        #print('loss is {0}'.format(eval_loss(w, b, x_list, gt_y_list)))
    t2=time.time()
    print('w:{0},b:{1}'.format(w,b))
    print('loss is {0}'.format(eval_loss(w, b, x_list, gt_y_list)))
    print('time is {0}'.format(t2-t1))

def gen_sample_data():
    w = random.randint(0, 10) + random.random()# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 10000
    x_list = random.randint(0,100,num_samples) * random.random(num_samples)
    y_list = w * x_list + b + random.random(num_samples) * random.randint(-1,1,num_samples)
    print("ans:",w,b)
    return x_list, y_list, w, b

def run():
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_list, y_list, 500, lr, max_iter)

if __name__ == '__main__':
    run()
