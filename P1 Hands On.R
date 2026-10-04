set.seed(260)

library(tidyverse)
library(rsample)
library(FNN)
library(rpart)


# CONFIGURAÇÃO ----


K_EXT <- 10   # folds da validação cruzada externa (avaliação)
K_INT <- 10   # folds da validação cruzada interna (tunagem)


# DADOS ----


# https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer

salario <- read_csv(
  "/home/arthur/Downloads/archive(1)/Salary Data.csv"
)

head(salario)

# Filtrando nível de educação
salario$`Education Level` |> table()

salario <- salario |>
  filter(`Education Level` == "Bachelor's")


# ANÁLISE DESCRITIVA----
median(salario$`Years of Experience`)
median(salario$Salary)

hist(salario$Salary, main = "Salário anual em US dollars")
hist(salario$`Years of Experience`, main = "Anos de experiência")
hist(salario$Age, main = "Idade")

plot(salario$Salary, salario$Age,
     xlab = "Salário", ylab = "Idade")

plot(salario$Salary, salario$`Years of Experience`,
     xlab = "Salário", ylab = "Anos de experiência")

plot(salario$`Years of Experience`, salario$Salary, 
     xlab = "Anos de experiência", ylab = "Salário")

salario <- salario |>
  select(`Years of Experience`, Salary) |>
  drop_na()



# FUNÇÕES AUXILIARES----

metricas <- function(y, pred) {
  c(
    RMSE = sqrt(mean((y - pred)^2)),
    MAE  = mean(abs(y - pred)),
    R2   = 1 - sum((y - pred)^2) / sum((y - mean(y))^2)
  )
}

ridge <- function(x, y, lambda) {
  X <- cbind(1, x)
  I <- diag(2)
  I[1, 1] <- 0
  solve(t(X) %*% X + lambda * I) %*% t(X) %*% y
}

prever_loess <- function(x_treino, y_treino, x_novo, span) {
  
  modelo <- tryCatch(
    suppressWarnings(
      loess(y ~ x, data = data.frame(x = x_treino, y = y_treino),
            span = span)
    ),
    error = function(e) NULL
  )
  if (is.null(modelo)) return(rep(NA_real_, length(x_novo)))
  
  x_limitado <- pmin(pmax(x_novo, min(x_treino)), max(x_treino))
  
  suppressWarnings(
    predict(modelo, newdata = data.frame(x = x_limitado))
  )
}

plot_grid_search <- function(lista_curvas, grid, escolhidos,
                             xlab, main, log = "", cor = "steelblue") {
  
  mat   <- sapply(lista_curvas, function(d) d$RMSE)
  media <- rowMeans(mat, na.rm = TRUE)
  
  plot(
    grid, media,
    type = "n",
    log = log,
    ylim = range(mat, na.rm = TRUE),
    xlab = xlab,
    ylab = "RMSE (CV interna)",
    main = main
  )
  
  for (f in seq_len(ncol(mat))) {
    lines(grid, mat[, f], col = "grey70")
  }
  
  lines(grid, media, col = cor, lwd = 3, type = "b", pch = 16, cex = 0.6)
  
  abline(v = escolhidos, lty = 2, col = "grey40")
  abline(v = grid[which.min(media)], lty = 1, col = "red", lwd = 2)
  
  legend(
    "topright",
    legend = c("Folds externos", "Média", "Escolhido por fold",
               "Melhor da média"),
    col = c("grey70", cor, "grey40", "red"),
    lty = c(1, 1, 2, 1),
    lwd = c(1, 3, 1, 2),
    bty = "n",
    cex = 0.8
  )
}


# VALIDAÇÃO CRUZADA EXTERNA----

folds_externos <- vfold_cv(salario, v = K_EXT)


# GRID DE HIPERPARÂMETROS----

grid_lambda <- 10^seq(0, 5, length.out = 100)   # Ridge
grid_k      <- seq(2, 50, by = 1)               # KNN
grid_span   <- seq(0.25, 1, by = 0.05)          # LOWESS
grid_depth  <- 1:10                             # Árvore


# OBJETOS PARA ARMAZENAR RESULTADOS----

modelos <- c("Linear", "Ridge", "KNN", "LOWESS", "Arvore")

resultados_externos <- data.frame(Fold = 1:K_EXT)
for (m in modelos) {
  for (met in c("RMSE", "MAE", "R2")) {
    resultados_externos[[paste0(m, "_", met)]] <- NA_real_
  }
}

curvas_ridge  <- vector("list", K_EXT)
curvas_knn    <- vector("list", K_EXT)
curvas_lowess <- vector("list", K_EXT)
curvas_arvore <- vector("list", K_EXT)

escolhidos <- data.frame(
  lambda = rep(NA_real_, K_EXT),
  k      = NA_real_,
  span   = NA_real_,
  depth  = NA_real_
)


# LOOP DOS FOLDS EXTERNOS----

for (j in seq_along(folds_externos$splits)) {
  
  cat("\n==============================\n")
  cat("FOLD EXTERNO:", j, "de", K_EXT, "\n")
  cat("==============================\n")
  
  treino_externo <- analysis(folds_externos$splits[[j]])
  validacao_externa <- assessment(folds_externos$splits[[j]])
  
  # Alvo e preditor da validação EXTERNA (nunca sobrescrever!)
  y_val_externa <- validacao_externa$Salary
  x_val_externa <- validacao_externa$`Years of Experience`
  
  x_tr_externo <- treino_externo$`Years of Experience`
  y_tr_externo <- treino_externo$Salary
  
  # Folds internos (compartilhados por todos os modelos)
  folds_internos <- vfold_cv(treino_externo, v = K_INT)
  
  
  # 1. REGRESSÃO LINEAR----
  
  modelo_linear <- lm(Salary ~ `Years of Experience`,
                      data = treino_externo)
  
  pred_linear <- predict(modelo_linear, newdata = validacao_externa)
  met_linear <- metricas(y_val_externa, pred_linear)
  
  
  # 2. RIDGE----
  
  resultados_ridge <- data.frame(lambda = grid_lambda, RMSE = NA_real_)
  
  for (i in seq_along(grid_lambda)) {
    
    RMSE_folds <- c()
    
    for (f in seq_along(folds_internos$splits)) {
      
      tr_int  <- analysis(folds_internos$splits[[f]])
      val_int <- assessment(folds_internos$splits[[f]])
      
      beta <- ridge(tr_int$`Years of Experience`,
                    tr_int$Salary,
                    grid_lambda[i])
      
      y_pred <- beta[1] + beta[2] * val_int$`Years of Experience`
      
      RMSE_folds <- c(
        RMSE_folds,
        sqrt(mean((val_int$Salary - y_pred)^2))
      )
    }
    
    resultados_ridge$RMSE[i] <- mean(RMSE_folds)
  }
  
  melhor_lambda <- resultados_ridge$lambda[which.min(resultados_ridge$RMSE)]
  curvas_ridge[[j]] <- resultados_ridge
  
  beta_final <- ridge(x_tr_externo, y_tr_externo, melhor_lambda)
  pred_ridge <- beta_final[1] + beta_final[2] * x_val_externa
  met_ridge <- metricas(y_val_externa, pred_ridge)
  
  
  # 3. KNN----
  
  resultados_knn <- data.frame(k = grid_k, RMSE = NA_real_)
  
  for (i in seq_along(grid_k)) {
    
    RMSE_folds <- c()
    
    for (f in seq_along(folds_internos$splits)) {
      
      tr_int  <- analysis(folds_internos$splits[[f]])
      val_int <- assessment(folds_internos$splits[[f]])
      
      y_pred <- knn.reg(
        train = matrix(tr_int$`Years of Experience`, ncol = 1),
        test  = matrix(val_int$`Years of Experience`, ncol = 1),
        y     = tr_int$Salary,
        k     = grid_k[i]
      )$pred
      
      RMSE_folds <- c(
        RMSE_folds,
        sqrt(mean((val_int$Salary - y_pred)^2))
      )
    }
    
    resultados_knn$RMSE[i] <- mean(RMSE_folds)
  }
  
  melhor_k <- resultados_knn$k[which.min(resultados_knn$RMSE)]
  curvas_knn[[j]] <- resultados_knn
  
  pred_knn <- knn.reg(
    train = matrix(x_tr_externo, ncol = 1),
    test  = matrix(x_val_externa, ncol = 1),
    y     = y_tr_externo,
    k     = melhor_k
  )$pred
  
  met_knn <- metricas(y_val_externa, pred_knn)
  
  
  # 4. LOWESS----
  
  resultados_lowess <- data.frame(span = grid_span, RMSE = NA_real_)
  
  for (i in seq_along(grid_span)) {
    
    RMSE_folds <- c()
    
    for (f in seq_along(folds_internos$splits)) {
      
      tr_int  <- analysis(folds_internos$splits[[f]])
      val_int <- assessment(folds_internos$splits[[f]])
      
      y_pred <- prever_loess(
        tr_int$`Years of Experience`,
        tr_int$Salary,
        val_int$`Years of Experience`,
        grid_span[i]
      )
      
      if (any(is.na(y_pred))) next
      
      RMSE_folds <- c(
        RMSE_folds,
        sqrt(mean((val_int$Salary - y_pred)^2))
      )
    }
    
    
    if (length(RMSE_folds) == length(folds_internos$splits)) {
      resultados_lowess$RMSE[i] <- mean(RMSE_folds)
    }
  }
  
  if (all(is.na(resultados_lowess$RMSE))) {
    melhor_span <- 1
  } else {
    melhor_span <- resultados_lowess$span[which.min(resultados_lowess$RMSE)]
  }
  curvas_lowess[[j]] <- resultados_lowess
  
  pred_lowess <- prever_loess(x_tr_externo, y_tr_externo,
                              x_val_externa, melhor_span)
  met_lowess <- metricas(y_val_externa, pred_lowess)
  
  
  # 5. ÁRVORE DE DECISÃO----
  
  resultados_arvore <- data.frame(maxdepth = grid_depth, RMSE = NA_real_)
  
  for (i in seq_along(grid_depth)) {
    
    RMSE_folds <- c()
    
    for (f in seq_along(folds_internos$splits)) {
      
      tr_int  <- analysis(folds_internos$splits[[f]])
      val_int <- assessment(folds_internos$splits[[f]])
      
      modelo_arvore <- rpart(
        Salary ~ `Years of Experience`,
        data = tr_int,
        method = "anova",
        control = rpart.control(maxdepth = grid_depth[i])
      )
      
      y_pred <- predict(modelo_arvore, newdata = val_int)
      
      RMSE_folds <- c(
        RMSE_folds,
        sqrt(mean((val_int$Salary - y_pred)^2))
      )
    }
    
    resultados_arvore$RMSE[i] <- mean(RMSE_folds)
  }
  
  melhor_depth <- resultados_arvore$maxdepth[which.min(resultados_arvore$RMSE)]
  curvas_arvore[[j]] <- resultados_arvore
  
  modelo_arvore_final <- rpart(
    Salary ~ `Years of Experience`,
    data = treino_externo,
    method = "anova",
    control = rpart.control(maxdepth = melhor_depth)
  )
  
  pred_arvore <- predict(modelo_arvore_final, newdata = validacao_externa)
  met_arvore <- metricas(y_val_externa, pred_arvore)
  
  
  # GUARDAR RESULTADOS DO FOLD EXTERNO----
  
  lista_met <- list(
    Linear = met_linear,
    Ridge  = met_ridge,
    KNN    = met_knn,
    LOWESS = met_lowess,
    Arvore = met_arvore
  )
  
  for (m in modelos) {
    for (met in c("RMSE", "MAE", "R2")) {
      resultados_externos[[paste0(m, "_", met)]][j] <- lista_met[[m]][[met]]
    }
  }
  
  escolhidos$lambda[j] <- melhor_lambda
  escolhidos$k[j]      <- melhor_k
  escolhidos$span[j]   <- melhor_span
  escolhidos$depth[j]  <- melhor_depth
  
  cat("Lambda escolhido:", melhor_lambda, "\n")
  cat("K escolhido:", melhor_k, "\n")
  cat("Span escolhido:", melhor_span, "\n")
  cat("Profundidade escolhida:", melhor_depth, "\n")
}


# RESULTADOS DOS FOLDS EXTERNOS----

resultados_externos
escolhidos


# COMPARAÇÃO FINAL----

comparacao <- data.frame(
  Modelo = c("Regressão Linear", "Ridge", "KNN", "LOWESS", "Árvore"),
  RMSE = sapply(modelos, \(m) mean(resultados_externos[[paste0(m, "_RMSE")]])),
  MAE  = sapply(modelos, \(m) mean(resultados_externos[[paste0(m, "_MAE")]])),
  R2   = sapply(modelos, \(m) mean(resultados_externos[[paste0(m, "_R2")]]))
)
rownames(comparacao) <- NULL

# Ordenar pelo RMSE
comparacao <- comparacao[order(comparacao$RMSE), ]

comparacao


# GRÁFICOS DE GRID SEARCH----
par(mfrow = c(2, 2))

plot_grid_search(
  curvas_ridge, grid_lambda, escolhidos$lambda,
  xlab = "Lambda", main = "Grid Search - Ridge",
  log = "x", cor = "steelblue"
)

plot_grid_search(
  curvas_knn, grid_k, escolhidos$k,
  xlab = "Número de vizinhos (K)", main = "Grid Search - KNN",
  cor = "darkgreen"
)

plot_grid_search(
  curvas_lowess, grid_span, escolhidos$span,
  xlab = "Span", main = "Grid Search - LOWESS",
  cor = "darkorange"
)

plot_grid_search(
  curvas_arvore, grid_depth, escolhidos$depth,
  xlab = "Profundidade máxima", main = "Grid Search - Árvore de Decisão",
  cor = "purple"
)


# ÁRVORE FINAL (ajustada em todos os dados)----

profundidade_final <- as.numeric(
  names(which.max(table(escolhidos$depth)))
)

arvore_final <- rpart(
  Salary ~ `Years of Experience`,
  data = salario,
  method = "anova",
  control = rpart.control(maxdepth = profundidade_final)
)

par(mar = c(2, 2, 2, 2))

plot(
  arvore_final,
  uniform = TRUE,
  margin = 0.1,
  compress = TRUE,
  main = paste("Árvore de decisão (profundidade =", profundidade_final, ")")
)

text(
  arvore_final,
  use.n = TRUE,
  all = TRUE,
  cex = 0.7
)

par(mar = c(5, 4, 4, 2) + 0.1)  


# COMPARAÇÃO DOS MODELOS: RMSE POR FOLD EXTERNO----

rmse_long <- resultados_externos |>
  select(Fold, ends_with("_RMSE")) |>
  pivot_longer(
    -Fold,
    names_to = "Modelo",
    values_to = "RMSE",
    names_pattern = "(.*)_RMSE"
  )

boxplot(
  RMSE ~ Modelo,
  data = rmse_long,
  col = "lightblue",
  xlab = "Modelo",
  ylab = "RMSE (folds externos)",
  main = "Distribuição do RMSE na validação cruzada externa"
)

stripchart(
  RMSE ~ Modelo,
  data = rmse_long,
  vertical = TRUE,
  method = "jitter",
  pch = 16,
  add = TRUE
)