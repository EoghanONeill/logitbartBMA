#' @title ITE Predictions (in-sample) using bartBMA and the method described by Hill (2011)
#'
#' @description This function produces ITE Predictions (in-sample) using bartBMA and the method described by Hill (2011).
#' @param x_covariates Covaraite matrix for training bartBMA.
#' @param z_train treatment vector for traiing bartBMA.
#' @param y_train outcome vector for training bartBMA.
#' @param a This is a parameter that influences the variance of terminal node parameter values. Default value a=3.
#' @param nu This is a hyperparameter in the distribution of the variance of the error term. THe inverse of the variance is distributed as Gamma (nu/2, nu*lambda/2). Default value nu=3.
#' @param sigquant Calibration quantile for the inverse chi-squared prior on the variance of the error term.
#' @param c This determines the size of Occam's Window
#' @param pen This is a parameter used by the Pruned Exact Linear Time Algorithm when finding changepoints. Default value pen=12.
#' @param num_cp This is a number between 0 and 100 that determines the proportion of changepoints proposed by the changepoint detection algorithm to keep when growing trees. Default num_cp=20.
#' @param x.test Test data covariate matrix. Default x.test=matrix(0.0,0,0).
#' @param num_rounds Number of trees. (Maximum number of trees in a sum-of-tree model). Default num_rounds=5.
#' @param alpha Parameter in prior probability of tree node splitting. Default alpha=0.95
#' @param beta Parameter in prior probability of tree node splitting. Default beta=1
#' @param split_rule_node Binary variable. If equals 1, then find a new set of potential splitting points via a changepoint algorithm after adding each split to a tree. If equals zero, use the same set of potential split points for all splits in a tree. Default split_rule_node=0.
#' @param gridpoint Binary variable. If equals 1, then a grid search changepoint detection algorithm will be used. If equals 0, then the Pruned Exact Linear Time (PELT) changepoint detection algorithm will be used (Killick et al. 2012). Default gridpoint=0.
#' @param maxOWsize Maximum number of models to keep in Occam's window. Default maxOWsize=100.
#' @param num_splits Maximum number of splits in a tree
#' @param gridsize This integer determines the size of the grid across which to search if gridpoint=1 when finding changepoints for constructing trees.
#' @param zero_split Binary variable. If equals 1, then zero split trees can be included in a sum-of-trees model. If equals zero, then only trees with at least one split can be included in a sum-of-trees model.
#' @param only_max_num_trees Binary variable. If equals 1, then only sum-of-trees models containing the maximum number of trees, num_rounds, are selected. If equals 0, then sum-of-trees models containing less than num_rounds trees can be selected. The default is only_max_num_trees=1.
#' @param min_num_obs_for_split This integer determines the minimum number of observations in a (parent) tree node for the algorithm to consider potential splits of the node.
#' @param min_num_obs_after_split This integer determines the minimum number of observations in a child node resulting from a split in order for a split to occur. If the left or right chikd node has less than this number of observations, then the split can not occur.
#' @export
#' @return A list of length 2. The first element is A vector of Individual Treatment Effect Estimates. The second element is a bartBMA object (i.e. the trained BART-BMA model).
#' @examples
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
#' example_output <- ITEs_bartBMA(x_covariates = x_covs_train,
#'                                z_train = ztrain,
#'                                y_train = ytrain)

ITEs_logitbartBMA<-function(x_covariates,z_train ,y_train,
                       a=3,nu=3,sigquant=0.9,c=1000,
                       pen=12,num_cp=20,x.test=matrix(0.0,0,0),
                       num_rounds=5,alpha=0.95,beta=2,split_rule_node=0,
                       gridpoint=1,maxOWsize=100,num_splits=5,gridsize=10,zero_split=1,only_max_num_trees=1,
                       min_num_obs_for_split=2, min_num_obs_after_split=2,
                       exact_residuals=1,
                       spike_tree=0, s_t_hyperprior=1, p_s_t=0.5, a_s_t=1,b_s_t=3,
                       lambda_poisson=10,less_greedy=0,
                       calc_intervals = 1,
                       ncores = 1,
                       root_alg_precision=0.00001,
                       maxit=300,
                       eps_f = 1e-8,
                       eps_g = 1e-5,
                       lower_prob=0.025,
                       upper_prob=0.975,
                       num_iter=100,
                       include_cate_intervals=1){



  x_train <- cbind(z_train,x_covariates)

  object <- logitbartBMA(x.train = x_train,y.train = y_train,
    a=a,nu=nu,sigquant=sigquant,c=c,
    pen=pen,num_cp=num_cp,x.test=x.test,
    num_rounds=num_rounds,alpha=alpha,beta=beta,split_rule_node=split_rule_node,
    gridpoint=gridpoint,maxOWsize=maxOWsize,num_splits=num_splits,
    gridsize=gridsize,zero_split=zero_split,only_max_num_trees=only_max_num_trees,
    min_num_obs_for_split=min_num_obs_for_split, min_num_obs_after_split=min_num_obs_after_split,
    exact_residuals=exact_residuals,
    spike_tree=spike_tree, s_t_hyperprior=s_t_hyperprior, p_s_t=p_s_t, a_s_t=a_s_t,b_s_t=b_s_t,
    lambda_poisson=lambda_poisson,less_greedy=less_greedy,
    calc_intervals = calc_intervals,
    ncores = ncores,
    root_alg_precision=root_alg_precision,
    maxit=maxit,
    eps_f = eps_f,
    eps_g = eps_g,
    lower_prob=lower_prob,
    upper_prob=upper_prob)



    ret <- pred_ints_lbbma_ITE_insamp_par(object$sumoftrees,
                                          object$obs_to_termNodesMatrix,
                                          object$response,
                                          object$model_probs,
                                          object$nrowTrain,
                                          object$a,
                                          object$sigma,
                                          0,
                                          lower_prob,
                                          upper_prob,
                                          ncores,
                                          root_alg_precision,
                                          maxit,
                                          eps_f,
                                          eps_g,
                                          x_train,
                                          num_iter,
                                          include_cate_intervals)




  #PI<-apply(draws_from_mixture,2,function(x)quantile(x,probs=c(l_quant,0.5,u_quant)))



  #each row is a vector drawn from the mixture distribution





  if(include_cate_intervals==1){

    names(ret)<-c("ITE_estimates",
                  "ITE_intervals",
                  "CATE_estimate",
                  "CATE_Interval")





  }else{
    names(ret)<-c("ITE_estimates",
                  "ITE_intervals",
                  "CATE_estimate")



  }

  ret2 <- list()
  ret2$ITE_est_objs<- ret
  ret2$bbma_object <- object
  ret2
}
