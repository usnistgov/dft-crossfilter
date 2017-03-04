#x11(width=1, height=1)       # X11 Plot dimensions
library(minpack.lm)          # Load the minpack.lm package

mydata = read.csv("crossfilter_app/Rdata.csv")  # Read CSV data file
x<-mydata$Kpts_atom         # Select the kpoints atom density
y<-mydata$P

dy<-mydata$P_err            # Select the ground state energy E0

l  = length(x)              # Length of data set for down selection
l1 = 3                      # Number of data points to be removed from beginning
l2 = 0                      # Number of data points to be removed from end

precs = list()
predicts = list()
extrapolates = c()
errors = c()
orders = c()

result <- tryCatch({
# 1th order modified Pade
m<-nlsLM(y~(a1*x)/(b0+x), start = list(a1=y[l], b0=1), subset = l1:l-l2, weights = x^3 * dy^-1)
m1=m
#par(mfrow=c(1,2))
extrapol = coef(m)[1]
#plot(x,y, lty=2,col="black",lwd=3)                   # Plot data and fit
#lines(x[l1:l-l2],predict(m),lty=2,col="red",lwd=3)

#plot(x, y-extrapol, col="black", ylim=c(-0.5, 0.5))
#lines(x[l1:l-l2], predict(m)-extrapol, type="l", lty=2, col="red",lwd=3, log="")
#lines(x[l1:l-l2],abs(predict(m)-extrapol),lty=2,col="red",lwd=3)
summary(m)                  # Report summary of fit, values and error bars

orders = append(orders, 1)
extrapolates = append(extrapolates, extrapol)
errors= append(errors, coef(summary(m))[1, "Std. Error"])

precs[[length(precs)+1]] <- predict(m)-extrapol
predicts[[length(predicts)+1]] <- predict(m)

#precs = append(precs, predict(m)-extrapol)
#predicts = append(predicts, predict(m))


# 2nd order Pade
m<-nlsLM(y~(a1*x+a2*x^2)/(b0+b1*x+x^2), start = list(a1 = coef(m)[1], a2 = coef(m)[1], b0 = coef(m)[2], b1 = 1.0), subset = l1:l-l2, weights = x^3 * dy^-1)
m2=m

#par(mfrow=c(1,2))
extrapol = coef(m)[2]
#plot(x,y, lty=2,col="blue",lwd=3)                   # Plot data and fit
#lines(x[l1:l-l2],predict(m),lty=2,col="red",lwd=3)

#plot(x, abs(y-extrapol), log="y")
#lines(x[l1:l-l2],predict(m)-extrapol,lty=2,col="green",lwd=3)
summary(m)                  # Report summary of fit, values and error bars

orders = append(orders, 2)
extrapolates = append(extrapolates, extrapol)
errors= append(errors, coef(summary(m))[2, "Std. Error"])
precs[[length(precs)+1]] <- predict(m)-extrapol
predicts[[length(predicts)+1]] <- predict(m)


# 3rd order Pade
m<-nlsLM(y~(a0+a1*x+a2*x^2+a3*x^3)/(b0+b1*x+b2*x^2+x^3), start = list(a0 = coef(m)[1], a1 = coef(m)[2], a2 = coef(m)[3], a3 = coef(m)[3], b0 = coef(m)[4], b1 = coef(m)[5], b2 = 1.0), subset = l1:l-l2, weights = x^3 * dy^-1)
m3=m
#par(mfrow=c(1,2))
extrapol = coef(m)[4]
#plot(x,y, lty=2,col="blue",lwd=3)                   # Plot data and fit
#lines(x[l1:l-l2],predict(m),lty=2,col="red",lwd=3)

#plot(x, abs(y-extrapol), log="y")
#lines(x[l1:l-l2],abs(predict(m)-extrapol),lty=2,col="blue",lwd=3)
summary(m)                  # Report summary of fit, values and error bars


orders = append(orders, 3)
extrapolates = append(extrapolates, extrapol)
errors= append(errors, coef(summary(m))[4, "Std. Error"])
precs[[length(precs)+1]] <- predict(m)-extrapol
predicts[[length(predicts)+1]] <- predict(m)


# 4th order Pade
m<-nlsLM(y~(a0+a1*x+a2*x^2+a3*x^3+a4*x^4)/(b0+b1*x+b2*x^2+b3*x^3+x^4), start = list(a0 = coef(m)[1], a1 = coef(m)[2], a2 = coef(m)[3], a3 = coef(m)[4], a4=coef(m)[4], b0 = coef(m)[5], b1 = coef(m)[6], b2 = coef(m)[7], b3 = 1.0), subset = l1:l-l2, weights = x^3 * dy^-1)
m4=m
#par(mfrow=c(1,2))
extrapol = coef(m)[5]
#plot(x,y, lty=2,col="blue",lwd=3)                   # Plot data and fit
#lines(x[l1:l-l2],predict(m),lty=2,col="red",lwd=3)

#plot(x, abs(y-extrapol), log="y")
#lines(x[l1:l-l2],abs(predict(m)-extrapol),lty=2,col="black",lwd=3)
summary(m)                  # Report summary of fit, values and error bars


orders = append(orders, 4)
extrapolates = append(extrapolates, extrapol)
errors= append(errors, coef(summary(m))[4, "Std. Error"])

precs[[length(precs)+1]] <- predict(m)-extrapol
predicts[[length(predicts)+1]] <- predict(m)


print (paste(predict(m)-extrapol), length(predict(m)-extrapol))
anova(m1,m2,m3,m4)
}, error = function(err){

print(paste("error found at ", err))
}
)

#print(paste(result))

summary(m)

#std_err= coef(summary(m))[order+1, "Std. Error"]

#print(paste(std_err, order, extrapolate))
#print (paste(predict(m)-extrapol))

#print (paste(predict(m)))
#print(paste(precs))
predicts
errors
#length(extrapolates)
output  = data.frame(Order = orders, Extrapolate = extrapolates, Error=errors)# Precisions = list(precs), Predicts=list(predicts))
write.csv(output, 'Result.csv')


pade1 <- function(x) {
    pade <- (a0+a1*x)/(b0+x)
    return(pade)
}

a0 = y[l]
a1 = y[l]
b0 = 1
plot(x,y, lty=2,col="red",lwd=3)
lines(x,pade1(x))
