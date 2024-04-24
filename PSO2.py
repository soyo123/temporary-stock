import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from random import sample, randint, random

################# Begin Import the train data1(training data japan1.csv) ##################
with open('training data japan1.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the title row（if have）
    next(reader)
    # Define the index of columns
    columns_of_interest = [2]  # the index begin from 0
    # Define a list
    str_array = []
    for row in reader:
        selected_data = [row[col] for col in columns_of_interest]  # choose the columns
        # Put the item into the list
        str_array.append(selected_data[0])
# Change the string item of list into float
train_japan1 = [float(s) for s in str_array]
# print(train_japan1)
#####################################END#########################################

################# Begin Import the train data2(actual data)(training data japan2.csv) ##############
with open('training data japan2.csv', 'r', newline='') as csvfile2:
    reader2 = csv.reader(csvfile2)
    # Skip the title row（if have）
    next(reader2)
    # Define the index of columns
    columns_of_interest2 = [2]  # the index begin from 0
    # Define a list
    str_array2 = []
    for row2 in reader2:
        selected_data2 = [row2[col] for col in columns_of_interest2]  # choose the columns
        # Put the item into the list
        str_array2.append(selected_data2[0])
# Change the string item of list into float
train_japan2 = [float(s) for s in str_array2]
# print(train_japan2)
#####################################END#########################################



x_max = 1 # The max dimension
x_min = 0 # The min dimension
N=30 # Number of population
D=3 # Dimension

# Generating the population
x = np.random.rand(N, D) * (x_max - x_min) + x_min
# print(x)

############################# Begin Holt-winters model ##############################
series=train_japan1 #Define the seasonal data list

# count=len(series)
# print(count)

#initial_trend(series, 12)
slen=12
#Generating the initial trend
def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

#initial_seasonal_components(series, 12)
slen=12
#Genearating the initial seasonal components
def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

#triple_exponential_smoothing(series, 12, 0.716, 0.029, 0.993, 24)
slen=12
n_preds=24
# Define the holt-winters model
def holt_model(series, slen, alpha, beta, gamma, n_preds):
    # The holt-winters model 
    def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
        result = []
        seasonals = initial_seasonal_components(series, slen)
        for i in range(len(series)+n_preds):
            if i == 0: # initial values
                smooth = series[0]
                trend = initial_trend(series, slen)
                result.append(series[0])
                continue
            if i >= len(series): # we are forecasting
                m = i - len(series) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
                trend = beta * (smooth-last_smooth) + (1-beta)*trend
                seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])
        return result


    # 创建数据
    y = triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds)
    # print(y)
    # count2=len(y)
    # print(count2)

    #Caculate the MAPE
    # Define the dataset as python lists 
    forecastAll = triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds)
    array_forecast=[]
    for i in range(72,96):
        array_forecast.append(forecastAll[i])

    # print(len(array_forecast))

    # Define the dataset as python lists 
    actual   = train_japan2 
    forecast = array_forecast

    # Consider a list APE to store the 
    # APE value for each of the records in dataset 
    APE = [] 

    # Iterate over the list values 
    for day in range(24): 

        # Calculate percentage error 
        per_err = (actual[day] - forecast[day]) / actual[day] 

        # Take absolute value of 
        # the percentage error (APE) 
        per_err = abs(per_err) 

        # Append it to the APE list 
        APE.append(per_err) 

    # Calculate the MAPE 
    MAPE = sum(APE)/len(APE) 

    # Print the MAPE value and percentage 
    # print(f''' 
    # MAPE   : { round(MAPE, 2) } 
    # MAPE % : { round(MAPE*100, 2) } % 
    # ''')
    return MAPE
#############################END Holt-winters model ##############################

# Generating the inital pbest for PSO (PSO:X0...X15)
p_best=[]
for i in range(0,15):
    # alpha=0.716
    alpha=x[i,0]
    # beta=0.029、
    beta=x[i,1]
    # gamma=0.993
    gamma=x[i,2]
    # print(alpha)
    # print(beta)
    # print(gamma)
    p_best.append(holt_model(series, slen, alpha, beta, gamma, n_preds))################ get the pbest

# print(p_best)


################### PSO algorithm ###############################
def psoAlgorithm(w, c1 , c2, x, N, D, slen, n_preds):


    # 设置字体和设置负号
    # matplotlib.rc("font", family="KaiTi")
    # matplotlib.rcParams["axes.unicode_minus"] = False
    # 初始化种群，群体规模，每个粒子的速度和规模
    # N = 100 # 种群数目
    # D = 3 # 维度
    T = 20 # 最大迭代次数
    c1 = c2 = 1.5 # 个体学习因子与群体学习因子
    w_max = 0.8 # 权重系数最大值
    w_min = 0.4 # 权重系数最小值
    x_max = 4 # 每个维度最大取值范围，如果每个维度不一样，那么可以写一个数组，下面代码依次需要改变
    x_min = -4 # 同上
    v_max = 1 # 每个维度粒子的最大速度
    v_min = -1 # 每个维度粒子的最小速度


    # 定义适应度函数
    def func(series, slen, x, n_preds):
        print(x)
        alpha=x[0]
        beta=x[1]
        gamma=x[2]
        return holt_model(series, slen, alpha, beta, gamma, n_preds)


    # 初始化种群个体
    # x = np.random.rand(N, D) * (x_max - x_min) + x_min # 初始化每个粒子的位置
    v = np.random.rand(N, D) * (v_max - v_min) + v_min # 初始化每个粒子的速度

    # 初始化个体最优位置和最优值
    p = x # 用来存储每一个粒子的历史最优位置
    p_best = np.ones((N, 1))  # 每行存储的是最优值
    for i in range(N): # 初始化每个粒子的最优值，此时就是把位置带进去，把适应度值计算出来
        p_best[i] = func(series, slen,x[i, :],n_preds) 

    # 初始化全局最优位置和全局最优值
    g_best = 100 #设置真的全局最优值
    gb = np.ones(T) # 用于记录每一次迭代的全局最优值
    x_best = np.ones(D) # 用于存储最优粒子的取值

    # 按照公式依次迭代直到满足精度或者迭代次数
    for i in range(T):
        for j in range(N):
            # 个更新个体最优值和全局最优值
            if p_best[j] > func(series, slen,x[j,:],n_preds):
                p_best[j] = func(series, slen,x[j,:],n_preds)
                p[j,:] = x[j,:].copy()
            # p_best[j] = func(x[j,:]) if func(x[j,:]) < p_best[j] else p_best[j]
            # 更新全局最优值
            if g_best > p_best[j]:
                g_best = p_best[j]
                x_best = x[j,:].copy()   # 一定要加copy，否则后面x[j,:]更新也会将x_best更新
            # 计算动态惯性权重
            w = w_max - (w_max - w_min) * i / T
            # 更新位置和速度
            v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p[j, :] - x[j, :]) + c2 * np.random.rand(1) * (x_best - x[j, :])
            x[j, :] = x[j, :] + v[j, :]
            # 边界条件处理
            for ii in range(D):
                if (v[j, ii] > v_max) or (v[j, ii] < v_min):
                    v[j, ii] = v_min + np.random.rand(1) * (v_max - v_min)
                if (x[j, ii] > x_max) or (x[j, ii] < x_min):
                    x[j, ii] = x_min + np.random.rand(1) * (x_max - x_min)
        # 记录历代全局最优值
        gb[i] = g_best
    # print("最优值为", gb[T - 1],"最优位置为",x_best)
    # plt.plot(range(T),gb)
    # plt.xlabel("迭代次数")
    # plt.ylabel("适应度值")
    # plt.title("适应度进化曲线")
    plt.show()
    return gb[T - 1],x_best

w=1
c1=1.5
c2=1.5
m=psoAlgorithm(w, c1 , c2, x, N, D,slen, n_preds)
print(m[0])