import numpy as np
import time
from numpy import random
from functools import reduce
from matplotlib import pyplot as plot
random.seed(123)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def inference(w, b, x1, x2):        # inference, test, predict, same thing. Run model after training
    pred_y = sigmoid(w * x1 + b - x2)
    return pred_y

# I will have to set 1e-6 here to avoid log(0)
def eval_loss(w, b, x1_list, x2_list, gt_y_list):
    h=inference(w,b,x1_list,x2_list)
    h[h<1e-6]=1e-6
    h[(1-h)<1e-6]=1-(1e-6)
    return -np.mean(gt_y_list*np.log(h)+(1-gt_y_list)*np.log(1-h))

# note this is the same as linear regression
def cal_step_gradient(batch_x1_list, batch_x2_list, batch_gt_y_list, w, b, lr):
    batch_size = len(batch_x1_list)
    pred_y = inference(w,b,batch_x1_list,batch_x2_list)
    avg_dw = np.mean((pred_y - batch_gt_y_list) * batch_x1_list)
    avg_db = np.mean((pred_y - batch_gt_y_list))
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b

def train(x1_list, x2_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    num_samples = len(x1_list)
    t1=time.time()
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x1_list), batch_size)
        w, b = cal_step_gradient(x1_list[batch_idxs], x2_list[batch_idxs], gt_y_list[batch_idxs], w, b, lr)
        #print('w:{0}, b:{1}'.format(w, b))
        #print('loss is {0}'.format(eval_loss(w, b, x1_list, x2_list, gt_y_list)))
    t2=time.time()
    print('w:{0},b:{1}'.format(w,b))
    print('loss is {0}'.format(eval_loss(w, b, x1_list, x2_list, gt_y_list)))
    print('time is {0}'.format(t2-t1))

def gen_sample_data():
    w = random.randint(0, 10) + random.random()# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 10000
    x1_list = random.randint(0,100,num_samples) * (random.random(num_samples)-0.5)
    x2_list = random.randint(0,100,num_samples) * (random.random(num_samples)-0.5)
    idx = (w*x1_list+b) > x2_list
    y_list = np.zeros(num_samples)
    y_list[idx]=1.0
    # introduce some error
    p=random.random(num_samples)
    idx= (p > 0.95)
    y_list[idx] = 1.0 - y_list[idx]
    print("ans:",w,b)
    # if you want to make a plot
    #idx=(y_list==1) 
    #plt.plot(x1_list[idx],x2_list[idx],'x');plt.plot(x1_list[~idx],x2_list[~idx],'o');plt.show() 
    return x1_list, x2_list, y_list, w, b

def run():
    x1_list, x2_list, y_list, w, b = gen_sample_data()
    lr = 1.0
    max_iter = 10000
    train(x1_list, x2_list, y_list, 500, lr, max_iter)

if __name__ == '__main__':
    run()
