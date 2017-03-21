library(minpack.lm)          # Load the minpack.lm package

mydata <- read.csv("crossfilter_app/Rdata.csv")  # Read CSV data file
x= mydata$Kpt         # Select the kpoints atom density
y= mydata$P

            # Select the ground state energy E0

l  <- length(x)   # Length of data set for down selection
l1 <- 3   # Number of data points to be removed from beginning
l2 <- 0                      # Number of data points to be removed from end

precs <- list()
predicts <- list()
extrapolates <- c()
errors <- c()
orders <- c()
coeffs <- c()

#result <- tryCatch({
# 1th order modified Pade
m <- nlsLM(y~ (a1*x)/(b0+x), start = list(a1 = y[l], b0 = 1), subset = l1:l-l2, weights = x^3)
m1 <- m
extrapol <- coef(m)[1]
coeffs <- append(coeffs, c(coef(m1)[1],coef(m1)[2],coef(m1)[3]))
summary(m)                  # Report summary of fit, values and error bars

orders <- append(orders, 1)
extrapolates <- append(extrapolates, extrapol)
errors<- append(errors, coef(summary(m))[1, "Std. Error"])

#precs[[length(precs)+1]] <- predict(m)-extrapol
#predicts[[length(predicts)+1]] <- predict(m)

#precs <- append(precs, predict(m)-extrapol)
#predicts <- append(predicts, predict(m))


# 2nd order Pade
m<-nlsLM(y~(a1*x+a2*x^2)/(b0+b1*x+x^2), start = list(a1 = coef(m)[1], a2 = coef(m)[1], b0 = coef(m)[2], b1 = 1.0), subset = l1:l-l2, weights = x^3)
m2<-m

extrapol <- coef(m)[2]

summary(m)                  # Report summary of fit, values and error bars

orders <- append(orders, 2)
extrapolates <- append(extrapolates, extrapol)
errors<- append(errors, coef(summary(m))[2, "Std. Error"])
#precs[[length(precs)+1]] <- predict(m)-extrapol
#predicts[[length(predicts)+1]] <- predict(m)
coeffs <- append(coeffs, c(coef(m2)[1],coef(m2)[2],coef(m2)[3],coef(m2)[4]))

# 3rd order Pade
#m<-nlsLM(y~(a0+a1*x+a2*x^2+a3*x^3) / (b0+b1*x+b2*x^2+x^3), start = list(a0 = coef(m)[1], a1 = coef(m)[2], a2 = coef(m)[3], a3 = coef(m)[3], b0 = coef(m)[4], b1 = coef(m)[5], b2 = 1.0), subset = l1:l-l2, weights = x^3)
#m3<-m
#extrapol <- coef(m)[4]
#summary(m)                  # Report summary of fit, values and error bars


#orders <- append(orders, 3)
#extrapolates <- append(extrapolates, extrapol)
#errors<- append(errors, coef(summary(m))[4, "Std. Error"])
#coeffs <- append(coeffs, coeff(m))

# 4th order Pade
#m<-nlsLM(y~(a0+a1*x+a2*x^2+a3*x^3+a4*x^4)/(b0+b1*x+b2*x^2+b3*x^3+x^4), start = list(a0 = coef(m)[1], a1 = coef(m)[2], a2 = coef(m)[3], a3 = coef(m)[4], a4 = coef(m)[4], b0 = coef(m)[5], b1 = coef(m)[6], b2 = coef(m)[7], b3 = 1.0), subset = l1:l-l2, weights = x^3)
#m4<-m
#extrapol <- coef(m)[5]
#summary(m)                  # Report summary of fit, values and error bars


#orders <- append(orders, 4)
#extrapolates <- append(extrapolates, extrapol)
#errors<- append(errors, coef(summary(m))[4, "Std. Error"])
#coeffs <- append(coeffs, coeff(m))

#print (paste(predict(m)-extrapol), length(predict(m)-extrapol))
#anova(m1,m2,m3,m4)
#},
#error <- function(result){
#print(paste("error found at ", result))
#}
#)

summary(m)

#std_err<- coef(summary(m))[order+1, "Std. Error"]

#print(paste(std_err, order, extrapolate))
#print (paste(predict(m)-extrapol))

#print (paste(predict(m)))
#print(paste(precs))
#predicts
#errors
typeof(coeffs)
typeof(c(predict(m2)))
#length(extrapolates)
predicts_plot <- data.frame(Preds <- c(predict(m2)))
output  <- data.frame(Order <- orders, Extrapolate <- extrapolates, Error<-errors)# Precisions <- list(precs), Predicts<-list(predicts))
write.csv(output, 'crossfilter_app/Result.csv')
write.csv(predicts_plot, 'crossfilter_app/Predicts.csv')

pade1 <- function(x) {
    a1 <- coef(m1)[1]
    b1 <- coef(m1)[2]
    pade <- (a1*x)/(b1+x)
    return(pade)
}

pade2 <- function(x) {
    a1 <- coef(m2)[1]
    a2 <- coef(m2)[2]
    b0 <- coef(m2)[3]
    b1 <- coef(m2)[4]
    pade <- (a1*x+a2*x^2)/(b0+b1*x+x^2)
    return(pade)
}

x_plot <- seq(from=1000, to=max(x)+1000000, by=100)
pade2(x_plot)
pade1_plot <- data.frame(Px <- x_plot, Py <- pade1(x_plot))
pade2_plot <- data.frame(Px <- x_plot, Py <- pade2(x_plot))
write.csv(pade1_plot, 'crossfilter_app/Pade1.csv')
write.csv(pade2_plot, 'crossfilter_app/Pade2.csv')

a0 <- y[l]
a1 <- y[l]
b0 <- 1
#plot(x,y, lty<-2,col<-"red",lwd<-3)
#lines(x,pade1(x))
