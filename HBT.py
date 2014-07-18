# -*- coding: utf-8 -*-
"""
Created on Tue Jul 01 15:56:58 2014

@author: Annina
"""

from pylab import *
from scipy.integrate import quad
from math import factorial

d=50
D=5500
x=frange(-100,100,0.01)
r1=sqrt((d/2.+x)**2+D**2)
r2=sqrt((-d/2.+x)**2+D**2)
k1=pi/2
k2=3*pi/2
r=0.05
g=pi*r**2*(1+exp(-1j*(k1*r1+k2*r2))*cos(k1*r1-k2*r2))

#plot(x,g)

#show()
#K=2*pi/lambda
k=pi/2.5
th1=1
th2=2
dx=299
#th1=sin(theta1)
#th2=sin(theta2)

#th= theta

a1=frange(0,20,0.01)
b1=a1+dx

a2=50.
b2=a2+dx

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

G11=b1-a1
G22=b2-a2
G12=(1/(k*th1))**2*(exp(1j*k*(b1-b2)*th1)-exp(1j*k*(a1-b2)*th1)-exp(1j*k*(b1-a2)*th1)+exp(1j*k*(a1-a2)*th1)) + (1/(k*th2))**2*((exp(1j*k*(b1-b2)*th2))-exp(1j*k*(a1-b2)*th2)-exp(1j*k*(b1-a2)*th2)+exp(1j*k*(a1-a2)*th2))
G21=(1/(k*th1))**2*(exp(-1j*k*(b1-b2)*th1)-exp(-1j*k*(a1-b2)*th1)-exp(-1j*k*(b1-a2)*th1)+exp(-1j*k*(a1-a2)*th1)) + (1/(k*th2))**2*((exp(-1j*k*(b1-b2)*th2))-exp(-1j*k*(a1-b2)*th2)-exp(-1j*k*(b1-a2)*th2)+exp(-1j*k*(a1-a2)*th2))
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

z=frange(1,51,0.025)

#Scheiben (im Fourierraum)
#Saha: det's in a u,v plane, Grössenordnung u = 10^7*lambda, darum r1 und r2 so gross...
r1 =1000
r2=1000
area1 = pi*r1**2
area2 = pi*r2**2
#print area1, area2
det1=area1
det2=area2

#eine Scheibe aus 1 und rundherum 0 (im Otsraum)
source=
            

#Eine Scheibe nicht nur aus 1 sondern unregelmässig
#sources=

#detectors * source  und fourier davon
#sdet=source*ifftshift(ifft(ifftshift(det1)))*ifftshift(ifft(ifftshift(det2)))
#sdets=sources*ifftshift(ifft(ifftshift(det1)))*ifftshift(ifft(ifftshift(det2)))

#result=fftshift(fft(fftshift(sdet)))
#results=fftshift(fft(fftshift(sdets)))

#G2=abs(result)**2
#G2s=abs(results)**2

#plot(G2)




show()