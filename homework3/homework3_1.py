# Q for teacher: in cal_step_gradient, it seems the for-loop is faster than reduce, do you know why?
import numpy as np
import random
import time
from functools import reduce
np.random.seed(123)
random.seed(123)

def inference(w, b, x):        # inference, test, predict, same thing. Run model after training
    pred_y = w * x + b
    return pred_y

def eval_loss(w, b, x_list, gt_y_list):
    avg_loss = 0.0
    for i in range(len(x_list)):
        avg_loss += 0.5 * (w * x_list[i] + b - gt_y_list[i]) ** 2    # loss function
    avg_loss /= len(gt_y_list)
    return avg_loss

def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dw = diff * x
    db = diff
    return [dw, db]

def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    avg_dw, avg_db = 0.0, 0.0
    batch_size = len(batch_x_list)
    dtmp = [gradient(inference(w,b,x),y,x) for x,y in zip(batch_x_list,batch_gt_y_list)]
    #avg_dw, avg_db = reduce(lambda x,y:(x[0]+y[0],x[1]+y[1]), dtmp) # this line is 10% slower than for-loop
    for dd in dtmp:
        avg_dw += dd[0]
        avg_db += dd[1]
    avg_dw /= batch_size
    avg_db /= batch_size
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
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        #print('w:{0}, b:{1}'.format(w, b))
        #print('loss is {0}'.format(eval_loss(w, b, x_list, gt_y_list)))
    t2=time.time()
    print(w,b,t2-t1)

def gen_sample_data():
    w = random.randint(0, 10) + random.random()# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 10000
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w, b

def run():
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_list, y_list, 500, lr, max_iter)

if __name__ == '__main__':
    run()
