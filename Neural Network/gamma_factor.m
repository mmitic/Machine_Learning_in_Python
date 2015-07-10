
x=0:5000
y=1./exp(-x./2500)

normalize=0.5+(0.95-0.5)*(y-min(y))./(max(y)-min(y))
plot(x, normalize)

