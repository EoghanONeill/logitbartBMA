#' @title Estimate ITEs and obtain credible intervals (in-sample or out-of-sample).
#'
#' @description This function takes a set of sum of tree models obtained from ITEs_bartBMA, and then estimates ITEs, and obtains prediction intervals.
#' @param object Output from ITEs_bartBMA of class ITE_ests.bartBMA.
#' @param l_quant Lower quantile of credible intervals for the ITEs, CATT, CATNT.
#' @param u_quant Upper quantile of credible intervals for the ITEs, CATT, CATNT.
#' @param newdata Test data for which predictions are to be produced. Default = NULL. If NULL, then produces prediction intervals for training data if no test data was used in producing the bartBMA object, or produces prediction intervals for the original test data if test data was used in producing the bartBMA object.
#' @param update_resids Option for whether to update the partial residuals in the gibbs sampler. If equal to 1, updates partial residuals, if equal to zero, does not update partial residuals. The defaullt setting is to update the partial residuals.
#' @param num_cores Number of cores used in parallel.
#' @param root_alg_precision The algorithm should obtain approximate bounds that are within the distance root_alg_precision of the true quantile for the chosen average of models.
#' @param training_data The training data matrix
#' @export
#' @return The output is a list of length 4:
#' \item{ITE_intervals}{A 3 by n matrix, where n is the number of observations. The first row gives the l_quant*100 quantiles of the individual treatment effects. The second row gives the medians of the ITEs. The third row gives the u_quant*100 quantiles of the ITEs.}
#' \item{ITE_estimates}{An n by 1 matrix containing the Individual Treatment Effect estimates.}
#' \item{CATE_estimate}{The Conditional Average Treatment Effect Estimates}
#' \item{CATE_Interval}{A 3 by 1 matrix. The first element is the l_quant*100 quantile of the CATE distribution, the second element is the median of the CATE distribution, and the thied element is the u_quant*100 quantile of the CATE distribution.}
#' @examples
#' #Example of BART-BMA for ITE estimation
#' # Applied to data simulations from Hahn et al. (2020, Bayesian Analysis)
#' # "Bayesian Regression Tree Models for Causal Inference: Regularization,
#' # Confounding, and Heterogeneous Effects
#' n <- 250
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- rbinom(n,1,0.5)
#' x5 <- as.factor(sample( LETTERS[1:3], n, replace=TRUE))
#'
#' p= 0
#' xnoise = matrix(rnorm(n*p), nrow=n)
#' x5A <- ifelse(x5== 'A',1,0)
#' x5B <- ifelse(x5== 'B',1,0)
#' x5C <- ifelse(x5== 'C',1,0)
#'
#' x_covs_train <- cbind(x1,x2,x3,x4,x5A,x5B,x5C,xnoise)
#'
#' #Treatment effect
#' #tautrain <- 3
#' tautrain <- 1+2*x_covs_train[,2]*x_covs_train[,4]
#'
#' #Prognostic function
#' mutrain <- 1 + 2*x_covs_train[,5] -1*x_covs_train[,6]-4*x_covs_train[,7] +
#' x_covs_train[,1]*x_covs_train[,3]
#' #mutrain <- -6 + 2*x_covs_train[,5] -1*x_covs_train[,6]-4*x_covs_train[,7] +
#' 6*abs(x_covs_train[,3]-1)
#' sd_mtrain <- sd(mutrain)
#' utrain <- runif(n)
#' #pitrain <- 0.8*pnorm((3*mutrain/sd_mtrain)-0.5*x_covs_train[,1])+0.05+utrain/10
#' pitrain <- 0.5
#' ztrain <- rbinom(n,1,pitrain)
#' ytrain <- mutrain + tautrain*ztrain
#' #pihattrain <- pbart(x_covs_train,ztrain )$prob.train.mean
#'
#' #set lower and upper quantiles for intervals
#' lbound <- 0.025
#' ubound <- 0.975
#'
#'
#' trained_bbma <- ITEs_bartBMA(x_covariates = x_covs_train,
#'                              z_train = ztrain,
#'                              y_train = ytrain)
#'
#' example_output <- ITEs_bartBMA_exact_par(trained_bbma[[2]],
#'                                          l_quant = lbound,
#'                                          u_quant= ubound,
#'                                          training_data = x_covs_train)

ITEs_logitbartBMA_from_object <-function(object,#min_possible,max_possible,
                                  l_quant,u_quant,newdata=NULL,update_resids=1,
                                  num_cores=1,
                                  root_alg_precision=0.00001,
                                  training_data,
                                  maxit=300,
                                  eps_f = 1e-8,
                                  eps_g = 1e-5,
                                  num_iter=100,
                                  include_cate_intervals=1){

  if(is.null(newdata) && object$is_test_data==1){
    #if test data specified separately
    ret<-pred_ints_lbbma_ITE_outsamp_par(object$sumoftrees,
                                    object$obs_to_termNodesMatrix,
                                    object$response,
                                    object$model_probs,
                                    object$nrowTrain,
                                    nrow(object$test_data),
                                    object$a,
                                    object$sigma,
                                    0,
                                    object$test_data,
                                    l_quant,
                                    u_quant,
                                    num_cores,
                                    root_alg_precision,
                                    maxit,
                                    eps_f,
                                    eps_g,
                                    training_data,
                                    num_iter,
                                    include_cate_intervals)

  }else{if(is.null(newdata) && object$is_test_data==0){
    #else return Pred Ints for training data

    ret <- pred_ints_lbbma_ITE_insamp_par(object$sumoftrees,
                                   object$obs_to_termNodesMatrix,
                                   object$response,
                                   object$model_probs,
                                   object$nrowTrain,
                                   object$a,
                                   object$sigma,
                                   0,
                                   l_quant,
                                   u_quant,
                                   num_cores,
                                   root_alg_precision,
                                   maxit,
                                   eps_f,
                                   eps_g,
                                   training_data,
                                   num_iter,
                                   include_cate_intervals)




  }else{
    #if test data included in call to object

    ret<-pred_ints_lbbma_ITE_outsamp_par(object$sumoftrees,
                                         object$obs_to_termNodesMatrix,
                                         object$response,
                                         object$model_probs,
                                         object$nrowTrain,
                                         nrow(newdata),
                                         object$a,
                                         object$sigma,
                                         0,
                                         newdata,
                                         l_quant,
                                         u_quant,
                                         num_cores,
                                         root_alg_precision,
                                         maxit,
                                         eps_f,
                                         eps_g,
                                         training_data,
                                         num_iter,
                                         include_cate_intervals)


  }}

  #PI<-apply(draws_from_mixture,2,function(x)quantile(x,probs=c(l_quant,0.5,u_quant)))



  #each row is a vector drawn from the mixture distribution





  if(include_cate_intervals==1){

    names(ret)<-c("ITE_estimates",
                  "ITE_intervals",
                  "CATE_estimate",
                  "CATE_Interval")

    ret



  }else{
    names(ret)<-c("ITE_estimates",
                  "ITE_intervals",
                  "CATE_estimate")

    ret

  }









}
