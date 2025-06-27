#====================================
#   Pytorch basico
#====================================
#   JOSÉ JULIO LOPEZ MARQUEZ
#   FUNDAMENTOS DE IA
#   ESFM IPN MARZO 2025
#====================================

#===========================
#   Modulo de pytorch
#===========================
import torch

#==============================================================
#  En pytorch todo esta basado en operaciones tensoriales
#==============================================================
#  Un tensor vive en Rn x Rm x Ro x Rp ..etc
#==============================================================

#===========================================
# Escalar vacio (trae basura)
#===========================================
x = torch.empty(1)  #scalar
print(x)

#====================
#  Vector en R3
#====================
x = torch.empty(3)
print(x)

#===========================
#  Tensor en R2XR3
#===========================
x = torch.empty(2, 3)
print(x)

#==============================
#  Tensor de R2XR2XR3
#==============================
x = torch.empty(2, 2, 3)
print(x)

#==============================================
# torch.rand(size): numeros aleatorios  [0,1]
#===========================================
#   Tensor de numeros aleatorios de R5xR3
#============================================
x = torch.rand(5, 3)
print(x)

#====================================
# torch.zeros(size) llenar con 0
# torch.ones(size) llenar con 1
#=================================
# Tensor de R5xR3 lleno con ceros
#=====================================
x = torch.zeros(5, 3)
print(x)

#=========================================
# Checar tamaño (lista con dimensiones)
#=========================================
print(x.size())

#=============================================
#  Checar tipo de datos (default es float32)
#=============================================
print(x.dtype)

#=========================================
#  Especificando tipo de datos
#=========================================
x = torch.zeros(5, 3, dtype = torch.float16)
print(x)
print(x.dtype)

#========================================
#  Construir vector con datos
#========================================
x = torch.tensor([5.5, 3])
print(x.size())

#==============================================
#  Vector optimizable (variables del gradiente)
#==============================================
x = torch.tensor([5.5, 3], requires_grad=True)

#========================
#  Suma de tensores
#========================
y = torch.rand(2, 2)
x = torch.rand(2, 2)
z = x + y
z = torch.add(x,y)
print(z)
y.add_(x)
print(y)

#============================
#  Resta de tensores
#=========================
z = x - y
z = torch.sub(x,y)
print(z)

#=================
# Multiplicacion
#================
z = x * y
z = torch.mul(x,y)
print(z)

#====================
#  Division
#====================
z = x / y
z = torch.div(x,y)
print(z)

#=====================
#  Rebanadas
#=======================
x = torch.rand(5,3)
print(x)
print(x[:, 0])  # todos los renglones, columna 0
print(x[1, :])  # renglon 1, todas las columnas
print(x[1, 1])  # elemento en (1 , 1)

#============================
# Valor del elemento en (1,1)
#============================
print(x[1,1].item())

#===================================
#  Cambiar forma con torch.view()
#===================================
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1,8)  # -1: se infiera de las otras dimensiones
#  si -1 pytorch determinara automaticamente el tamaño necesario
print(x.size(), y.size(), z.size())

#===================================================
#  Convertir un tensor en arreglo numpy y viceversa
#===================================================
a = torch.ones(5)
b = a.numpy()
print(b)
print(type(b))

#========================================
# Le suma 1 a todas las entradas
#========================================
a.add_(1)
print(a)
print(b)

#==========================
#  De numpy a torch
#==========================
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

#========================================
#  Le suma 1 a todas las entradas de a
#========================================
a += 1
print(a)
print(b)

#======================================
# De CPU a GPU (si hay cuda)
#======================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Tengo GPU" + str(device))
    y_d = torch.ones_like(x, device=device)
    x_d = x.to(device)
    z_d = x_d + y_d
    #=======================================================
    # z = z_d.numpy()  # numpy no maneja tensores en el GPU
    #=======================================================
    # de vuelta al CPU
    #===================
    z = z_d.to("cpu")
    z = z.numpy()
    print(z)