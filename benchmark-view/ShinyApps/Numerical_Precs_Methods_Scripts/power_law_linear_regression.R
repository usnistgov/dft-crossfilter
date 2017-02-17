mydata = read.csv('Rdata.csv')

x = mydata$X
y = mydata$Y

power_law <-lm('log(y)~log(x)')

fit_values = exp(coef(power_law)[1] + coef(power_law)[2] * log(x))
dat = data.frame(x=x, y=y,f=fit_values)
write.csv(dat, 'predicts.csv')

C = coef(power_law)[1]
C_err = coef(summary(power_law))[1, "Std. Error"] 
M = coef(power_law)[2]
M_err = coef(summary(power_law))[2, "Std. Error"]

datp = data.frame(C=C, C_err=C_err, M=M, M_err=M_err)

write.csv(datp, 'params.csv')
