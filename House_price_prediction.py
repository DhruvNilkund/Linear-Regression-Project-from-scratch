import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
mu=[]
std=[] 

def load_data(filename):
    df=pd.read_csv(filename,sep=",",index_col=False)
    df.columns=["housesize","room","price"]
    data=np.array(df,dtype=float)
    plot_data(data[:,:2],data[:,-1])
    normalize(data)
    return data[:,:2],data[:,-1] # first one is the first two columns of the dataset and the second part is the third column

def normalize(data):
    for i in range(0,data.shape[1]-1):
        data[:,i]=(data[:,i]-np.mean(data[:,i]))/np.std(data[:,i])
        mu.append(np.mean(data[:,i]))
        std.append(np.std(data[:, i])) 
        
def plot_data(x,y):
    plt.xlabel("house-size")
    plt.ylabel("price")
    plt.plot(x[:,0],y,'bo')
    plt.show()

#For linear regression our hypothesis is h(x)=theta0+theta1*x1+theta2*x2+.....
def h(x,theta):
    return np.matmul(x,theta) 

#Equation of cost function J(theta)= (1/2*m) * summation over ((h(x)-y)^2) from 1 to m 
def cost_function(x,y,theta):
    return ((h(x,theta)-y).T@(h(x,theta)-y))/(2*y.shape[0])

#Gradient descent in our concept is an optimization algorithm that aims to adjust the parameters in order to minimize the cost function(loss function)
#Since we have only two parameters here , our cost function is a function of theta0 and theta1 
# thetaj is thetaj - alpha*partiald(wrt(thetaj)J(theta0,theta1))
def gradient_descent(x,y,theta,learning_rate=0.1,num_epochs=10):
    m=x.shape[0]
    J_all=[]
    for _ in range(num_epochs):
        h_x=h(x,theta)
        cost_=(1/m)*(x.T@(h_x-y))
        theta=theta-learning_rate*cost_ 
        J_all.append(cost_function(x,y,theta))
    return theta,J_all

def test(theta,x):
    x[0]=(x[0]-mu[0])/std[0]
    x[1]=(x[1]-mu[1])/std[1] 
    y=theta[0]+theta[1]*x[0]+theta[2]*x[1] 
    print("Price of house: ",y)
    
x,y=load_data("C:/Users/User/Desktop/vs code projects/house_price_data.txt")
y=np.reshape(y,(46,1)) 
x=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.zeros((x.shape[1],1))
learning_rate=0.1
num_epochs=10 
theta,J_all=gradient_descent(x,y,theta,learning_rate,num_epochs)
J=cost_function(x,y,theta)
print("Cost: ",J)
print("Parameters: ",theta)

#for testing and plotting cost 
n_epochs=[]
jplot=[]
count=0 
for i in J_all:
    jplot.append(i[0][0])
    n_epochs.append(count)
    count+=1 
jplot=np.array(jplot) 
n_epochs=np.array(n_epochs)

test(theta,[1600,2])
