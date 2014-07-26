# -*- coding: utf-8 -*-
"""
Created on Tue Jul 01 15:56:58 2014

@author: Annina, David
"""

from pylab import *
from scipy.integrate import quad
from math import factorial
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt


#function which delivers a point-source: width/height x/y, point source in the middle with radius r pixels
def source2d(x,y,r):
    res = np.zeros(dtype=complex,shape=(x,y))
    y,x = np.ogrid[-x/2: x/2, -y/2: y/2]
    mask = (x)**2+(y)**2 <= r**2
    res[mask]=1
    '''    
    res = []
    newlin = []
    for i in range (-x/2, +x/2):
           for j in range (-y/2, +y/2):

             if(sqrt(i**2+j**2)<=r):
                newlin.append(1)
           else:
               newlin.append(0)
       res.append(newlin)
       newlin = []
    '''
    return res


#function which delivers a point-source: width/height x/y, point source in the middle with radius r pixels

def randsource2d(x,y,r):
    res = np.zeros(dtype=complex,shape=(x,y))
    y,x = np.ogrid[-x/2: x/2, -y/2: y/2]
    mask1 = (x)**2+(y)**2 <= r**2
    mask2 = (x-5)**2+(y+15)**2 <= (r-45)**2
    mask3 = (x+10)**2+(y-20)**2 <= (r-40)**2
    mask4 = (x-30)**2+(y+10)**2 <= (r-40)**2
    #res[mask]=np.round(np.random.rand(sum(mask))*2)
    res[mask1]=1
    res[mask2]=2
    res[mask3]=3
    res[mask4]=2
    '''    
    res = []
    newlin = []
    for i in range (-x/2, +x/2):
        for j in range (-y/2, +y/2):
            if(sqrt(i**2+j**2)<=r):
                newlin.append(randint(0,2))
            else:
                newlin.append(0)
        res.append(newlin)
        newlin = []
   ''' 
    return res

    

x,y = 200, 200

#wrapper for fftshift(ifft(fftshift(var))) to make editing easier
delx=10**-9

N=x
print delk
delk=2*pi/(N*delx)
print delk**2


def fwrp(var):
    return fftshift(fft(fftshift(var)))

def ifwrp(var):
    return ifftshift(ifft(ifftshift(var)))
        
def fwrp2(var):
    return fftshift(fft2(fftshift(var)))*(delx)**2

def ifwrp2(var):
    return ifftshift(ifft2(ifftshift(var)))*(delk)**2








#d=50
#D=5500
#x=frange(-100,100,0.01)
#r1=sqrt((d/2.+x)**2+D**2)
#r2=sqrt((-d/2.+x)**2+D**2)
#k1=pi/2
#k2=3*pi/2
#r=0.05
#g=pi*r**2*(1+exp(-1j*(k1*r1+k2*r2))*cos(k1*r1-k2*r2))

#plot(x,g)

#show()
#K=2*pi/lambda
#k=pi/2.5
#th1=1
#th2=2
#dx=299
#th1=sin(theta1)
#th2=sin(theta2)

#th= theta

#a1=frange(0,20,0.01)
#b1=a1+dx
#
#a2=50.
#b2=a2+dx

#det1 = [0.]*(len(a1)/2-dx/2)
#det1.extend([1./dx]*dx)
#det1.extend([0.]*(len(a1)/2-dx/2))
#
#det2 = [0.]*(len(a1)/2-dx/2)
#det2.extend([1./dx]*dx)
#det2.extend([0.]*(len(a1)/2-dx/2))

#det1 = [1]*dx
#det2 = [1]*(b2-a2)
#det.extend([0]*(len(a1)-dx))
#det2.extend([0]*(len(a1)-dx))
#print det

#G11=b1-a1
#G22=b2-a2
#G12=(1/(k*th1))**2*(exp(1j*k*(b1-b2)*th1)-exp(1j*k*(a1-b2)*th1)-exp(1j*k*(b1-a2)*th1)+exp(1j*k*(a1-a2)*th1)) + (1/(k*th2))**2*((exp(1j*k*(b1-b2)*th2))-exp(1j*k*(a1-b2)*th2)-exp(1j*k*(b1-a2)*th2)+exp(1j*k*(a1-a2)*th2))
#G21=(1/(k*th1))**2*(exp(-1j*k*(b1-b2)*th1)-exp(-1j*k*(a1-b2)*th1)-exp(-1j*k*(b1-a2)*th1)+exp(-1j*k*(a1-a2)*th1)) + (1/(k*th2))**2*((exp(-1j*k*(b1-b2)*th2))-exp(-1j*k*(a1-b2)*th2)-exp(-1j*k*(b1-a2)*th2)+exp(-1j*k*(a1-a2)*th2))
#G12=(exp(1j*k*(b1-b2)*th1)-exp(1j*k*(a1-b2)*th1)-exp(1j*k*(b1-a2)*th1)+exp(1j*k*(a1-a2)*th1)) + (1/(k*th2))**2*((exp(1j*k*(b1-b2)*th2))-exp(1j*k*(a1-b2)*th2)-exp(1j*k*(b1-a2)*th2)+exp(1j*k*(a1-a2)*th2))
#G21=(exp(-1j*k*(b1-b2)*th1)-exp(-1j*k*(a1-b2)*th1)-exp(-1j*k*(b1-a2)*th1)+exp(-1j*k*(a1-a2)*th1)) + (1/(k*th2))**2*((exp(-1j*k*(b1-b2)*th2))-exp(-1j*k*(a1-b2)*th2)-exp(-1j*k*(b1-a2)*th2)+exp(-1j*k*(a1-a2)*th2))

#G12_redux=(exp(1j*(k*(b1-b2)*(th1-th2)/2))+exp(-1j*(k*(b1-b2)*(th1-th2)/2)))*exp(1j*k*(b1-b2)*(th1+th2)/2)-(exp(1j*(k*(a1-b2)*(th1-th2)/2))+exp(-1j*(k*(a1-b2)*(th1-th2)/2)))*exp(1j*k*(a1-b2)*(th1+th2)/2)-(exp(1j*(k*(b1-a2)*(th1-th2)/2))+exp(-1j*(k*(b1-a2)*(th1-th2)/2)))*exp(1j*k*(b1-a2)*(th1+th2)/2)+(exp(1j*(k*(a1-a2)*(th1-th2)/2))+exp(-1j*(k*(a1-a2)*(th1-th2)/2)))*exp(1j*k*(a1-a2)*(th1+th2)/2)
#x=(b2+a2)/2 - (b1+a1)/2
#
#gg12 = exp(1j*k*(x)*th1)+exp(1j*k*(x)*th2)
#gg21 = exp(-1j*k*(x)*th1)+exp(-1j*k*(x)*th2)
#
#FFg = ifftshift(ifft(ifftshift(gg12)))
#res = FFg*ifftshift(ifft(ifftshift(det1)))*ifftshift(ifft(ifftshift(det2)))
#
#G2 = G12*G21
#G2 = G12_redux*G12_redux.conjugate()
#plot(x,abs(fftshift(fft(fftshift(res))))**2)
#plot(gg12.real)
#plot(x,G2)
#print G2

#G = 0
#gms = []
#for m in range (0,100):
#    gamma= (quad(lambda t: t**(m+1)*exp(-t), 0, Inf))[0]
#    #print 'gamma: ' + str(gamma)
#    gm = (-1)**m*(x/2)**(2*m+1)/(factorial(m)*gamma)
#    gms.append(gm)
#    print 'gm: ' + str(gm)
#    G = G + gm
#    print 'G:' + str(G)



#Z = randsource2d(200,200,50).T



#Scheiben (im Fourierraum)

det1=source2d(x,y,50)
det2 = det1
det3=det2

#eine Scheibe aus 1 und rundherum 0 (im Otsraum)
source = source2d(x,y,20)

#Eine Scheibe nicht nur aus 1 sondern unregelmässig
sources=randsource2d(x,y,50)


#detectors * source  und fourier davon
sdet=(source)*ifwrp2(det1)*ifwrp2(det2)
sdets=sources*ifwrp2(det1)*ifwrp2(det2)#*ifwrp2(det3)

result=fwrp2(sdet)
results=fwrp2(sdets)

G2=abs(result)**2
G2s=abs(results)**2

#norm
z=ifwrp2(fwrp2(source))/source
#z[abs(z)==inf] = np.nan
z[abs(z)==inf] = 0
print z[100]

# new figure
fig, ax = plt.subplots()
#plt.imshow(sources.real)
#plt.figure()
plt.imshow(results.real)
plt.show()
plt.colorbar()

