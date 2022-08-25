import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

columns = ["Class","Alcohol","Malic acid","Ash","Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
df = pd.read_csv('wine.data', names=columns)


"""
lr = learning rate
m = pendiente
b = intercepto
_o = old
_n = new
"""

# Constantes
lr = 0.01
x1 = np.array(df["Alcohol"])
x2 = np.array(df["Ash"])
y = np.array(df["Color intensity"])
n = len(y)
n_var = len(y)

# Valores iniciales
mo = np.array([-5,-20])
bo = 0.1

# Calcular m
def gradM1 (m, b):
  i=0
  suma = 0
  while(i<n):
    yh = m[0]*x1[i] + m[1]*x2[i] + b
    dfy = yh - y[i]
    dfyx = dfy * x1[i]
    suma = suma + dfyx
    i=i+1

  mn = (1/n)*suma*lr

  return mn


# Calcular m
def gradM2 (m, b):
  i=0
  suma = 0
  while(i<n):
    yh = m[0]*x1[i] + m[1]*x2[i] + b
    dfy = yh - y[i]
    dfyx = dfy * x2[i]
    suma = suma + dfyx
    i=i+1

  mn = (1/n)*suma*lr

  return mn

# Caalcular b
def gradB (m, b):
  i=0
  suma = 0
  while(i<n):
    yh = m[0]*x1[i] + m[1]*x2[i]+b
    dfy = yh - y[i]
    suma = suma + dfy
    i=i+1
    
  bn = (1/n)*suma*lr

  return bn

# Calcualar MSE
def MSE (m, b):
  i=0
  suma = 0
  while(i<n):
    df = y[i]-(m[0]*x1[i] +m[1]*x2[i] +b)
    df2 = df**2
    suma = suma + df2
    i=i+1
    

  return suma/n

eras = 600
j=0
mn=np.array([0.,0.])
MSE_arr = [MSE(mo, bo)]
print(mo, bo)
while(j<eras and MSE(mo,bo)!=0.0):
  mn[0] = mo[0] - gradM1(mo, bo)
  mn[1] = mo[1] - gradM2(mo, bo)
  bn = bo - gradB(mo, bo)

  mse = MSE(mn, bn)
  print("Era", j+1, "x1=", mn[0], " x2=", mn[1], " MSE=",mse)
  MSE_arr.append(mse)

  mo = mn
  mo = mn
  bo = bn

  j=j+1


 
xmse = np.array(range(0, j+1))
ymse = MSE_arr
plt.title("MSE conforme a las eras")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(xmse, ymse, color = "red", marker = "o", label = "MSE")
plt.legend()
plt.show()