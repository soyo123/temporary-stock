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
    # y = triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds)
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
############################# END Holt-winters model ##############################

################### Begin PSO algorithm ###############################
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
        # print(x)
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
    # plt.show()
    return gb[T - 1],x_best

# w=1
# c1=1.5
# c2=1.5
# m=psoAlgorithm(w, c1 , c2, x, N, D,slen, n_preds)
# print(m[0])
################### END PSO algorithm ###############################


################### Begin DE algorithm ###############################
def deAlgorithm(cr, x, N, D, slen, n_preds):
    
    nIter=20
    # uniRand = np.random.uniform
    def func(series, slen,xs,n_preds):
    # print(x)
        # print(xs)
        alpha=xs[0]
        # print(alpha)
        beta=xs[1]
        gamma=xs[2]
        # print(holt_model(series, slen, alpha, beta, gamma, n_preds))
        return holt_model(series, slen, alpha, beta, gamma, n_preds)
    
        # N为个体数；nDim为解维度；nIter为迭代次数
    def de(D, cr, func, nIter,x):
        # xs = [uniRand(*xRange, nDim) for _ in range(N)]
        # xs = [x for _ in range(N)]
        # xs = [x[i] for _ in range(N)]
        xs = []
        for i in range(0,len(x)):
            xs.append(x[i])
        # xs = x
        # print(x)
        for _ in range(nIter):
            xs = evolve(xs, cr, func)
        fs = [func(series, slen,x,n_preds) for x in xs]
        xBest = xs[np.argmin(fs)]
        msg = f"当前最优结果为{np.min(fs)}，参数为"
        msg += ", ".join([f"{x:.4f}" for x in xBest])
        # print(msg)
        return np.min(fs),[f"{x:.4f}" for x in xBest]
    
    # if __name__=="__main__":
    #     de(D,cr,func,nIter,x)
    
    def evolve(xs,cr,func):
        # 变异
        vs = []
        N = len(xs[0])
        for _ in range(len(xs)):
            # print(xs)
            x = xs.pop(0)
            r1, r2, r3 = sample(xs, 3)
            xs.append(x)
            vs.append(r1 + random()*(r2-r3))
        # 交叉
        us = []
        for i in range(len(xs)):
            us.append(vs[i] if random() < cr else xs[i])
            j = randint(0, N-1)
            us[i][j] = vs[i][j]
        # 选择（这里选的最小）
        # print(us)
        xNext = []
        for x,u in zip(xs, us):
            xNext.append(x if func(series, slen,x,n_preds)<func(series, slen,u,n_preds) else u)
        return xNext
    
    return de(D, cr, func, nIter,x)

# print(deAlgorithm(cr, x, N, D, slen, n_preds))

################### Hyper-heuristic(GA based) algorithm ###############################
DNA_SIZE = 3
POP_SIZE = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 50
# X_BOUND = [-3, 3]
# Y_BOUND = [-3, 3]

#Define the population range list
wlist=[0.4,0.5,0.6,0.7,0.8] #choose for w value
c1list=[0,1.0,2.0,3.0,4.0] #choose for c1 value
c2list=[0,1.0,2.0,3.0,4.0] #choose for c2 value
crlist=[0,0.5,1.0] #choose for cr value

# Generating the pop1 
w1 = np.random.choice(wlist)
c11 = np.random.choice(c1list)
c21 = np.random.choice(c2list)
cr1 = np.random.choice(crlist)
pop1 = [w1,c11,c21,cr1]
# print(pop1)

# Generating the pop2
w2 = np.random.choice(wlist)
c12 = np.random.choice(c1list)
c22 = np.random.choice(c2list)
cr2 = np.random.choice(crlist)
pop2 = [w2,c12,c22,cr2]

# Generating the pop3
w3 = np.random.choice(wlist)
c13 = np.random.choice(c1list)
c23 = np.random.choice(c2list)
cr3 = np.random.choice(crlist)
pop3 = [w3,c13,c23,cr3]

############### Compute the fitness ################

# Compute the fitness
def get_fitness(pop1,pop2,pop3,x,D,slen, n_preds):
    ################################################## 
    ######### Compute the fitness of group1 ##########
    ##################################################
    # Choose the x1 for PSO
    x1=[]
    for i in range(0,5):
        x1.append(x[i])
    x1=np.array(x1)
    # Caculate the psofitness1 by PSO
    w=pop1[0]
    c1=pop1[1]
    c2=pop1[2]
    N1=5
    psofitness1=psoAlgorithm(w, c1 , c2, x1, N1, D,slen, n_preds)
    # print(psofitness1[0])
    # Choose the x2 for DE
    x2=[]
    for i in range(5,10):
        x2.append(x[i])
    #Caculate the defitness1 by DE
    cr=pop1[3]
    N2=5
    defitness1=deAlgorithm(cr, x2, N2, D, slen, n_preds)
    # print(defitness1[0])
    # Compeare the fitness1 between PSO and DE
    # min_fitness1= min(num1, num2)
    # 初始化最小值和对应的x值
    min_fitness1 = float('inf')
    min_x1 = None
    # 循环遍历x值并比较函数值
    if psofitness1[0] <= defitness1[0]:
        min_fitness1 = psofitness1[0]
        min_x1 = psofitness1[1]
    else:
        min_fitness1 = defitness1[0]
        min_x1 = defitness1[1]
    ##################################################
    ######### Compute the fitness of group2 ##########
    ##################################################
    # Choose the x3 for PSO
    x3=[]
    for i in range(10,15):
        x3.append(x[i])
    x3=np.array(x3)
    # Caculate the psofitness2 by PSO
    w=pop2[0]
    c1=pop2[1]
    c2=pop2[2]
    N3=5
    psofitness2=psoAlgorithm(w, c1 , c2, x3, N3, D,slen, n_preds)
    # print(psofitness2[0])
    # Choose the x4 for DE
    x4=[]
    for i in range(15,20):
        x4.append(x[i])
    # Caculate the defitness2 by DE
    cr=pop2[3]
    N4=5
    defitness2=deAlgorithm(cr, x4, N4, D, slen, n_preds)
    # print(defitness2[0])
    # Compeare the fitness2 between PSO and DE
    # min_fitness1= min(num1, num2)
    # 初始化最小值和对应的x值
    min_fitness2 = float('inf')
    min_x2 = None
    # 循环遍历x值并比较函数值
    if psofitness2[0] <= defitness2[0]:
        min_fitness2 = psofitness2[0]
        min_x2 = psofitness2[1]
    else:
        min_fitness2 = defitness2[0]
        min_x2 = defitness2[1]
    ##################################################
    ######### Compute the fitness of group3 ##########
    ##################################################
    # Choose the x5 for PSO
    x5=[]
    for i in range(20,25):
        x5.append(x[i])
    x5=np.array(x5)
    # Caculate the psofitness3 by PSO
    w=pop3[0]
    c1=pop3[1]
    c2=pop3[2]
    N5=5
    psofitness3=psoAlgorithm(w, c1 , c2, x5, N5, D,slen, n_preds)
    # print(psofitness3[0])
    # Choose the x6 for DE
    x6=[]
    for i in range(25,30):
        x6.append(x[i])
    # Caculate the defitness3 by DE
    cr=pop3[3]
    N6=5
    defitness3=deAlgorithm(cr, x6, N6, D, slen, n_preds)
    # print(defitness3[0])
    # Compeare the fitness3 between PSO and DE
    # min_fitness1= min(num1, num2)
    # 初始化最小值和对应的x值
    min_fitness3 = float('inf')
    min_x3 = None
    # 循环遍历x值并比较函数值
    if psofitness3[0] <= defitness3[0]:
        min_fitness3 = psofitness3[0]
        min_x3 = psofitness3[1]
    else:
        min_fitness3 = defitness3[0]
        min_x3 = defitness3[1]
    # Generate the array for the fitness and min_x
    fitness=[min_fitness1,min_fitness2,min_fitness3 ]
    min_x=[min_x1,min_x2,min_x3]
    return fitness,min_x

# def plot_3d(ax):

# 	X = np.linspace(*X_BOUND, 100)
# 	Y = np.linspace(*Y_BOUND, 100)
# 	X,Y = np.meshgrid(X, Y)
# 	Z = F(X, Y)
# 	ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm)
# 	ax.set_zlim(-10,10)
# 	ax.set_xlabel('x')
# 	ax.set_ylabel('y')
# 	ax.set_zlabel('z')
# 	plt.pause(3)
# 	plt.show()

def crossover_and_mutation(pop, CROSSOVER_RATE = 0.8):
	new_pop = []
	for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
		child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
		if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
			mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
			cross_points = np.random.randint(low=0, high=DNA_SIZE)	#随机产生交叉的点
			child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
		mutation(child,father)	#每个后代有一定的机率发生变异
		new_pop.append(child)
                            
	return new_pop

def mutation(child, father,MUTATION_RATE=0.003):
	if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
		mutate_point = np.random.randint(0, DNA_SIZE)	#随机产生一个实数，代表要变异基因的位置
		child[mutate_point] = father[mutate_point] 	#将变异点的二进制为反转

def select(pop, fitness):    # nature selection wrt pop's fitness
    # 为了选择概率最小的元素，我们需要将概率数组 p 反转（或取反）
    # 注意：概率必须是正数，并且总和为1，因此我们需要对概率进行归一化
    p_normalized = (fitness)/(fitness.sum())  # 归一化概率
    p_min = 1 - p_normalized  # 计算每个元素不被选中的概率
    p_min_normalized = p_min / p_min.sum()  # 归一化不被选中的概率
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=p_min_normalized)
    return pop[idx]

def print_info(pop1,pop2,pop3,x,D,slen, n_preds):
    fitness = get_fitness(pop1,pop2,pop3,x,D,slen, n_preds)
    fitnessvalue = fitness[0]
    fitnessx = fitness[1]
    min_fitness_index = np.argmin(fitnessvalue)
    print("min_fitness:", fitnessvalue[min_fitness_index])
    print("Parameter of holt-winters:", fitnessx[min_fitness_index])
    # x,y = translateDNA(pop)
    print("Parameter of low-level heuristics:", pop[min_fitness_index])
    # print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


if __name__ == "__main__":
	# fig = plt.figure()
	# ax = Axes3D(fig)	
	# plt.ion()#将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
	# plot_3d(ax)

	# pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*2)) #matrix (POP_SIZE, DNA_SIZE)
    pop = [pop1,pop2,pop3]
    # pop = np.array(pop)
    for _ in range(N_GENERATIONS):#迭代N代
		# x,y = translateDNA(pop)
		# if 'sca' in locals(): 
		# 	sca.remove()
		# sca = ax.scatter(x, y, F(x,y), c='black', marker='o');plt.show();plt.pause(0.1)
     pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
    #  pop = crossover_and_mutation(pop, CROSSOVER_RATE)
		# F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
		# fitness = get_fitness(pop)
     fitness = get_fitness(pop1,pop2,pop3,x,D,slen, n_preds)
     fitnessvalue=fitness[0]
     fitnessvalue = np.array(fitnessvalue)
     pop = select(pop, fitnessvalue) #选择生成新的种群
	# 修改NumPy的打印选项以显示小数点后的零
    print_info(pop[0],pop[1],pop[2],x,D,slen, n_preds)
	# plt.ioff()
	# plot_3d(ax)