library(shiny)
library(rbokeh)
library(htmlwidgets)
source("helpers.R")
records <- read.csv("Rdata.csv")
app <- shinyApp(ui = fluidPage(
      titlePanel("Least Squares Regression"),

      sidebarLayout(
      	sidebarPanel("input dashboard",
              h5("Select the element to plot:"),

              sliderInput("points", "Number of points:",
                          min=1, max=35, value= c(0,35), step=1.0  ),

              selectInput("element", 
                           label="Element: ",
                           choices= c("Al",
                                      "W",
                                      "Ni"  ),
                         selected = "Al"),
              
              selectInput("regression",
                           label="Regression Model: ",
                           list("y~x",
                                "y~exp(-x)") )
                         
            ),
      	mainPanel("output plot and table",
                      img(src='MGI_Logo.png', height = 100, width = 100), 
                      textOutput("text1"),
                      textOutput("text2"),
                      rbokehOutput("plotb"),
                      tableOutput("rlm")
            )
    )
),
server = function(input,output){

    output$text1 <- renderText({
        paste("You have selected", input$element)
                              })

    output$text2 <- renderText({
        paste("You have selected first", input$points[1]-1, "points and last", length(records$Kpoint) - input$points[2] + 1, "points","total", input$points[1] -1 + length(records$Kpoint) - input$points[2], "out of", length(records$Kpoint))
                              })

    datk <- reactive({
                     st  <- input$points[1]
                     mid <- input$points[2]
                     end <- length(records$Kpoint)
                     append(records$Kpoint[1:st], records$Kpoint[mid:end])
                    })

    datp <-  reactive({
                     st  <- input$points[1]
                     mid <- input$points[2]
                     end <- length(records$Kpoint)
                     switch(input$element,
                    "Al" = append(records$Al[1:st], records$Al[mid:end]),
                    "W" = append(records$W[1:st], records$W[mid:end]),
                    "Ni" = append(records$Ni[1:st], records$Ni[mid:end]) )
                     })
                    

    output$plotb <- renderRbokeh({
       ## define the figure dimensions to be square, plot scatter function 
       figure(plot_width=200, plot_height=200, xlab="kpoints", ylab="Lattice constant") %>% ly_points( datk(), datp() )                      
                                })

    output$rlm <- renderTable({
       data_fit <- data.frame(x=datk(), y=datp())
       lmR      <- switch(input$regression,
                   "y~x" = lm("y~x", data_fit),
                   "y~exp(-x)" = lm("y~exp(-x)", data_fit) )
       lmResult <- summary(lmR)
       con <- confint(lmR, level=0.96)
       cbind(lmResult$coefficients, con)
                             })


  }
)
runApp(app)

