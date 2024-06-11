#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
#include<ctime>
#include <Rcpp.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>


using namespace std;
using namespace Rcpp;
using namespace arma;

const double log2pi = std::log(2.0 * M_PI);


//' @title
//' wpcaCpp
//' @description
//' an efficient function to find the
//'
//' @param Y is the observed n*p feature matrix.
//' @param nPCs is the number of the principle components.
//'
//' @export
//[[Rcpp::export]]
Rcpp::List wpcaCpp(const arma::mat& Y, const int& nPCs, const bool& weighted=true){
  mat U, V;
  vec s;
  mat PCs, loadings;
  svd_econ(U, s, V, Y);
  PCs = U.cols(0, nPCs-1) *diagmat(s.subvec(0, nPCs-1));
  loadings = V.cols(0, nPCs-1);
  mat dY = PCs*loadings.t() - Y;
  rowvec W = mean(dY % dY);
  if(weighted){
    svd_econ(U, s, V, Y*diagmat(1.0/ sqrt(W)));
    // vec s2 =  s % s;
    mat loadings_unscale = diagmat(sqrt(W)) * V.cols(0, nPCs-1);
    mat V1;
    vec s1;
    svd_econ(loadings, s1, V1, loadings_unscale);
    PCs = U.cols(0, nPCs-1) * diagmat(s.subvec(0, nPCs-1)) * V1 * diagmat(s1);
    dY = PCs*loadings.t() - Y;
    W = mean(dY % dY);
  }
  List output = List::create(
    Rcpp::Named("PCs") = PCs,
    Rcpp::Named("loadings") = loadings,
    Rcpp::Named("W") = W);

  return output;
}

//' @title
//' getneighborhood_fast
//' @description
//' an efficient function to find the neighborhood based on the matrix of position and a pre-defined cutoff
//'
//' @param x is a n-by-2 matrix of position.
//' @param radius is a threashold of Euclidean distance to decide whether a spot is an neighborhood of another spot. For example, if the Euclidean distance between spot A and B is less than cutoff, then A is taken as the neighbourhood of B.
//' @return A sparse matrix containing the neighbourhood
//'
//' @export
// [[Rcpp::export]]
arma::sp_umat getneighborhood_fast(const arma::mat y, float radius)	{
  int N = y.n_rows;
  arma::sp_umat D(N, N);
  float dis;
  uvec idx, idx2;
  for (int j = 0; j < N-1; ++j)
  {
    idx = find(abs(y(j,0) - y.col(0))<radius);
    idx2 = find(idx>j);
    int p = idx2.n_elem;
    for (int i = 0; i < p; ++i)
    {
      dis = norm(y.row(idx(idx2(i))) - y.row(j), 2);
      if (dis < radius){
        D(idx(idx2(i)),j) = 1;
        D(j,idx(idx2(i))) = 1;
      }
    }
  }
  return D;
}

sp_mat get_spNbs(ivec x, const sp_mat& Adj) {
  // row is for pixel.
  //output a sparse matrix, i-th row contains labels of neighbor_i.
  // Make const iterator
  arma::sp_mat::const_iterator start = Adj.begin();
  //arma::sp_mat::const_iterator end   = Adj.end();
  // Calculate number of nonzero points
  //int n = std::distance(start, end);
  int n = Adj.n_nonzero;
  //cout << "n=" << n << endl;
  //cout << "n=" << Adj.n_nonzero << endl;

  sp_mat spNbs(x.n_elem, x.n_elem);    //neiborhood state matrix, matched with Adj.

  arma::sp_mat::const_iterator it = start;
  for(int i = 0; i < n; ++i)
  {
    //temp(0) = it.row();
    //temp(1) = it.col();
    spNbs(it.row(), it.col()) = x(it.col());
    ++it; // increment
  }
  return spNbs;
}

arma::rowvec calXenergy2D_i(arma::ivec x, int i, const arma::sp_mat& Adj, int K, const arma::vec alpha, const double beta)	{
  arma::sp_mat spNbs = get_spNbs(x, Adj);
  arma::sp_mat spNbs_t = spNbs.t();  // transform spNbs to iterate by column.
  arma::rowvec Ux_i = zeros<arma::rowvec>(K);
  int k;
  for (k = 0; k < K; k++)
  {
    arma::sp_mat col(spNbs_t.col(i));
    double n_sameS = 0;
    int nn = col.n_nonzero;
    if (nn == 0)	{
      Ux_i(k) = alpha(k);
    } else {
      for (arma::sp_mat::iterator j = col.begin(); j != col.end(); ++j) {
        n_sameS += (*j) == (k+1);
      }
      Ux_i(k) = alpha(k) + beta * (nn - n_sameS)/2;
    }
  }
  return Ux_i;
}

arma::mat calXenergy2D_sp(arma::ivec x, const arma::sp_mat& Adj, int K, const arma::vec alpha, const double beta)	{
  int n = x.n_elem;
  arma::sp_mat spNbs = get_spNbs(x, Adj);
  arma::sp_mat spNbs_t = spNbs.t();  // transform spNbs to iterate by column.
  arma::mat Ux(n, K);
  int i, k;
  for (k = 0; k < K; k++)
  {
    for (i = 0; i < n; i++)
    {
      arma::sp_mat col(spNbs_t.col(i));
      double n_sameS = 0;

      int nn = col.n_nonzero;
      for (arma::sp_mat::iterator j = col.begin(); j != col.end(); ++j) {
        n_sameS += (*j) == (k+1);
      }
      Ux(i, k) = alpha(k) + beta * (nn - n_sameS)/2;
    }
  }

  arma::mat C_mat = normalise(exp(-Ux), 1, 1); // pseudo likelihood of x.
  Ux = -log(C_mat); // normalized Ux, this is the energy of x.

  return Ux;
}

vec dmvnrm(const arma::mat& Y, arma::mat& A, arma::vec& mu0, arma::mat& W, arma::vec mu, arma::mat& sigma, double lambda, bool logd = false, int cores = 1) {
  omp_set_num_threads(cores);
  double tol=1e-6;
  int n = Y.n_rows;
  int p = Y.n_cols;
  vec out(n);

  arma::mat omega(p,p);
  mat M_zero(p,p);


#pragma omp parallel for schedule(static)

  for (int i=0;i<n;i++) {
    vec nonzero(p, fill::ones);
    vec zero(p);
    vec y=conv_to<colvec>::from(Y.row(i));
    arma::uvec zero_index=find(abs(y)<tol);
    arma::uvec nonzero_index=find(abs(y)>=tol);
    nonzero.elem(zero_index)=zeros(zero_index.n_elem);
    int m=zero_index.n_elem;
    arma::mat W_nonzero(p-m,p-m);
    arma::vec y_nonzero=y.elem(find(abs(y)>=tol));


//    double constants=0;
    double distribution, constants = -sum(nonzero)/2 * log2pi;
    arma::mat W_inv=inv(diagmat(W));
    arma::mat S=W_inv-W_inv*A*inv(inv(sigma)+trans(A)*W_inv*A)*trans(A)*W_inv;

    if(m==0){
      distribution=0.5*(-log(det(S))+(trans(y-(A*mu+mu0))*S*(y-(A*mu+mu0))).eval()(0,0));
    }
    else{
    zero.elem(zero_index)=ones(zero_index.n_elem);
    W_nonzero.diag()=nonzeros(nonzero/diagvec(W));
//    W_nonzero.diag()=nonzero/diagvec(W);
    M_zero.diag()=zero;
//    vec omega_diag=M_zero*(2*lambda/(1+2*lambda*diagvec(W)));
////    vec omega_diag=M_zero*(2*lambda/(1+2*lambda*diagvec(W)))+diagvec(W_nonzero);
//    omega.diag()=omega_diag;

//    arma::mat Inv=inv(eye(p,p)+omega*A*sigma*trans(A));
//    double remainder=0.5*(trans(A*mu+mu0)*Inv*omega*(A*mu+mu0)).eval()(0,0);
////    double remainder=0.5*(trans(A*mu+mu0)*Inv*omega*(A*mu+mu0)-2*trans(y)*W_nonzero*Inv*(A*mu+mu0)+trans(y)*W_nonzero*y-trans(y)*W_nonzero*A*sigma*trans(A)*Inv*W_nonzero*y).eval()(0,0);

//    double distribution=0.5*(log(det(W))+log(det(sigma))+log(det(inv(diagmat(W))+2*lambda*M_zero))+log(det(trans(A)*omega*A+inv(sigma))))+remainder;

    //compute f(y'|x)
    vec mu_y=A*mu+mu0;
    mat sigma_y=W+A*sigma*trans(A);
    //compute f(y'plus|x)
    vec mu_yplus=mu_y.elem(nonzero_index);
    mat sigma_yplus=sigma_y.submat(nonzero_index,nonzero_index);
    //compute f(y'0|y'plus,x)
    arma::mat sigma_yplus_inv=W_nonzero-W_nonzero*A.rows(nonzero_index)*inv(sigma.i()+trans(A.rows(nonzero_index))*W_nonzero*A.rows(nonzero_index))*trans(A.rows(nonzero_index))*W_nonzero;
    vec mu_y0=mu_y.elem(zero_index)+sigma_y.submat(zero_index,nonzero_index)*sigma_yplus_inv*(y.elem(nonzero_index)-mu_yplus);
    mat sigma_y0=sigma_y.submat(zero_index,zero_index)-sigma_y.submat(zero_index,nonzero_index)*sigma_yplus_inv*sigma_y.submat(nonzero_index,zero_index);

     distribution=0.5*(log(det(eye(m,m)+2*lambda*sigma_y0))+log(det(sigma_yplus))+(trans(mu_y0)*inv(eye(m,m)+2*lambda*sigma_y0)*mu_y0*2*lambda+trans(y_nonzero-mu_yplus)*sigma_yplus_inv*(y_nonzero-mu_yplus)).eval()(0,0));
//    double distribution=0.5*(-log(det(S))+log(det(S+2*lambda*M_zero))+(trans(A*mu+mu0)*S*(S+2*lambda*M_zero).i()*2*lambda*M_zero*(A*mu+mu0)).eval()(0,0));
    }

    for(int j=0;j<(p-m);j++){
      constants=constants+log(1-exp(-lambda*y_nonzero(j)*y_nonzero(j)));
    }
    out(i)= constants-distribution;
  }
  if (logd==false) {
    out=exp(out);
  }
  return(out);
}

double obj_beta(const arma::ivec& y, const arma::mat& gam, const arma::sp_mat& Adj, int K, const arma::vec alpha, const double beta){
  mat Ux = calXenergy2D_sp(y, Adj, K, alpha, beta); // Uy was normalized, so there is no need to normalized Uy.
  return -accu(gam % Ux);
}

double obj_lambda(const mat Y, const mat val_zero, const mat gam, const double lambda){
  int K=gam.n_cols,n=gam.n_rows;
  double result=0;
  for(int k=0;k<K;k++){
    for(int i=0;i<n;i++){
      double val_nonzero;
      vec nonzero=nonzeros(Y.row(i));
      for(int j=0;j<nonzero.n_elem;j++)
        val_nonzero=val_nonzero+log(1-exp(-lambda*(nonzero(j)*nonzero(j))));
      result=result+gam(i,k)*(val_nonzero-lambda*val_zero(i,k));
    }
  }
  return result;
}

//ICM_step
Rcpp::List runICM_sp(const arma::mat& Y, arma::mat& A, arma::vec& mu0, arma::mat& W, arma::ivec& x,  arma::mat& mu, arma::cube& sigma, const arma::sp_mat& Adj, arma::vec alpha, double& beta, double& lambda, const int& maxIter_ICM){

  int n = Y.n_rows, K = mu.n_cols;
  int iter, k;
  // energy of Y

  arma::mat Uy(n, K);
  for (k = 0; k < K; k++)	{
    arma::vec mu_k = mu.col(k);
    arma::mat sigma_k = sigma.slice(k);
    Uy.col(k) = -dmvnrm(Y, A, mu0, W, conv_to<colvec>::from(mu_k), sigma_k, lambda, true);
  }
  if(Uy.has_nan()||Uy.has_inf())
    cout<<"Distribution has a NaN or infinite element"<<endl;

  arma::vec Energy(maxIter_ICM);
  Energy(0) = INFINITY;
  arma::mat Ux(n, K);
  arma::mat U(n, K);
  arma::vec Umin(n);
  arma::uvec x_u(n);

  //--------------------------------------------------------------------------------
  // ICM algrithm
  //--------------------------------------------------------------------------------
  int Iteration = 1;
  for (iter=1; iter<maxIter_ICM; iter++) {

    Ux = calXenergy2D_sp(x, Adj, K, alpha, beta);
    U = Uy + Ux;
    Umin = min(U, 1);
    x_u = index_min(U,1);
    x = conv_to<ivec>::from(x_u) + 1;

    Energy(iter) = sum(Umin);
    if (Energy(iter) - Energy(iter-1) > 1e-5) {
      cout << "diff Energy = " << Energy(iter) - Energy(iter-1) << endl;
      break;
    }

    if (Energy(iter-1) - Energy(iter) < 1e-5)
    {
      cout << "ICM Converged at Iteration = " << iter  << endl;
      break;
    }
  }

  if (iter == maxIter_ICM) {
    Iteration = iter-1;
  } else {
    Iteration = iter;
  }

  vec energy = Energy.subvec(1, Iteration);
  arma::mat pxgn = exp(-Ux);

  List output = List::create(
    Rcpp::Named("x") = x,
    Rcpp::Named("U") = U,
    Rcpp::Named("Uy") = Uy,
    Rcpp::Named("Ux") = Ux,
    Rcpp::Named("pxgn") = pxgn,
    Rcpp::Named("energy") = energy);

  return output;
}

//compute the (I+2λ*sigma_y0y0')^(-1)
Rcpp::List CalInv(mat A, uvec zero_index, uvec nonzero_index, vec sigma_diag, double lambda, mat L, mat sigma_k){
  int p=A.n_rows,D=A.n_cols;
  int m=zero_index.n_elem;
  arma::mat sigma00_prime_inv(m,m);
  arma::mat sigmaplusplus_inv((p-m),(p-m));
  arma::mat E(m,m),Eplus(p-m,p-m), A0, Aplus;

  E.diag()=1/(sigma_diag.elem(zero_index)+vec(m,fill::ones)/(2*lambda));
  A0=A.rows(zero_index);
  sigma00_prime_inv=E-E*A0*inv(sigma_k.i()+trans(A0)*E*A0)*trans(A0)*E;

  Eplus.diag()=1/sigma_diag.elem(nonzero_index);
  Aplus=A.rows(nonzero_index);
  if(p-m>D)
     sigmaplusplus_inv=Eplus-Eplus*Aplus*inv(sigma_k.i()+trans(Aplus)*Eplus*Aplus)*trans(Aplus)*Eplus;
  else
     sigmaplusplus_inv=(Aplus*sigma_k*trans(Aplus)+diagmat(sigma_diag.elem(nonzero_index))).i();

  //compute the (I+2λ*sigma_yy)^(-1) Y>0
  arma::mat inv_mat;
  arma::mat sigma0plus=A0*sigma_k*trans(Aplus);

  if(m==p)
    inv_mat=sigma00_prime_inv/(2*lambda);
  else if(m<D)
    inv_mat=inv(2*lambda*(eye(m,m)/(2*lambda)+A0*sigma_k*trans(A0)+diagmat(sigma_diag.elem(zero_index))-sigma0plus*sigmaplusplus_inv*trans(sigma0plus)));
  else inv_mat=(sigma00_prime_inv-sigma00_prime_inv*sigma0plus*(-sigmaplusplus_inv-sigmaplusplus_inv*Aplus*inv(inv(sigma_k*trans(A0)*sigma00_prime_inv*A0*sigma_k)-trans(Aplus)*sigmaplusplus_inv*Aplus)*trans(Aplus)*sigmaplusplus_inv)*trans(sigma0plus)*sigma00_prime_inv)/(2*lambda);

  List output = List::create(
    Rcpp::Named("inv_Sigma22") = sigmaplusplus_inv,
    Rcpp::Named("inv_sigma") = inv_mat);

  return output;
}

// compute P(Z,Y'_0|Y)
Rcpp::List calConditionalDistribution(const arma::vec& y,arma::mat& A, arma::vec& mu0, arma::mat& W, arma::mat& mu, arma::cube& sigma, double& lambda){
  int p=A.n_rows, D=A.n_cols, K=mu.n_cols;
  double tol=1e-6;
  arma::uvec nonzero_index=find(abs(y)>=tol);
  arma::uvec zero_index=find(abs(y)<tol);
  int m=zero_index.n_elem;

  //  arma::mat inv_sigma;
  arma::mat mu_c((D+m),K);
  arma::cube sigma_c((D+m),(D+m),K);

  arma::mat Sigma11((D+m),(D+m));
  arma::mat Sigma12((D+m),(p-m));
  arma::mat Sigma22((p-m),(p-m));
  //  arma::mat inv_Sigma22((p-m),(p-m));
  arma::mat sigma22(p,p);

  arma::mat L(D+m,D+m);
  L.submat(D,D,D+m-1,D+m-1)=eye(m,m);

  arma::mat mu_yy((D+m),K);
  arma::cube sigma_yy((D+m),(D+m),K);
  arma::mat M((D+m),(D+m));
  M.submat(0,0,(D-1),(D-1))=eye(D,D);

  for(int k=0;k<K;k++){

    Sigma11.submat(0,0,D-1,D-1)=sigma.slice(k);
    Sigma11.submat(D,0,D+m-1,D-1)=A.rows(zero_index)*sigma.slice(k);
    Sigma11.submat(0,D,D-1,D+m-1)=trans(A.rows(zero_index)*sigma.slice(k));

    sigma22=A*sigma.slice(k)*trans(A)+W;

    Sigma11.submat(D,D,D+m-1,D+m-1)=sigma22.submat(zero_index,zero_index);
    Sigma12.rows(0,D-1)=trans(A.rows(nonzero_index)*sigma.slice(k));
    Sigma12.rows(D,D+m-1)=sigma22.submat(zero_index,nonzero_index);

    List res=CalInv(A, zero_index, nonzero_index, diagvec(W), lambda, L, sigma.slice(k));

    arma::mat inv_Sigma22=res["inv_Sigma22"];
    mu_c.col(k).subvec(0,D-1)=mu.col(k);
    vec temp=A*mu.col(k)+mu0;
    mu_c.col(k).subvec(D,D+m-1)=temp.elem(zero_index);
    mu_c.col(k)=mu_c.col(k)+Sigma12*inv_Sigma22*(y.elem(nonzero_index)-temp.elem(nonzero_index));
    sigma_c.slice(k)=Sigma11-Sigma12*inv_Sigma22*trans(Sigma12);

    arma::mat inv_mat=res["inv_sigma"];

    M.submat(D,0,(D+m-1),(D-1))=-2*lambda*inv_mat*sigma_c.slice(k).submat(D,0,(D+m-1),(D-1));
    M.submat(D,D,(D+m-1),(D+m-1))=inv_mat;

    mu_yy.col(k)=mu_c.col(k)-2*lambda*sigma_c.slice(k)*M*L*mu_c.col(k);
    sigma_yy.slice(k)=sigma_c.slice(k)-2*lambda*sigma_c.slice(k)*M*L*sigma_c.slice(k);
  }

  List output = List::create(
    Rcpp::Named("num_zero")= m,
    Rcpp::Named("mu") = mu_yy,
    Rcpp::Named("sigma") = sigma_yy);

  return output;
}


mat runEstep(mat U)	{
  int n = U.n_rows, K = U.n_cols;
  mat gam = zeros<mat>(n, K);

  vec maxU=max(-U,1);
  U=(-U-repmat(maxU,1,K));
  vec loglik_more_vec=sum(exp(U),1);
  gam=exp(U)/repmat(loglik_more_vec,1,K);
  gam.replace(0,1e-6);

  return gam;
}

//' @title
//' ICMEM
//' @description
//' estimates the joint latent expectations in the E-step.
//'
//' @param Y is a N*p feature matrix
//' @param x_int is a vector of initial cluster label.
//' @param Adj is a matrix containing neighborhood information generated by getneighborhood_fast.
//' @param A_int is an initial p*D factor loading matrix.
//' @param mu0_int is an initial p*1 mean vector.
//' @param W_int is an initial p*p diagonal matrix.
//' @param mu_int is an initial mean vector. we often generated it by Gaussian mixture model.
//' @param theta_int is a initial precision matrix. we often generated it by Gaussian mixture model.
//' @param lambda_grid is a sequence of exponential decay parameter in the zero-inflation model.
//' @param alpha is a intercept.
//' @param beta_grid is a sequence of smoothing parameter that can be specified by user.
//' @param maxiter_ICM is the maximum iteration of ICM algorithm.
//' @param maxiter is the maximum iteration of EM algorithm.
//' @return a list.
//' The item 'x' is the clustering result.
//' The item 'gam' is the posterior probability matrix.
//' The item 'A' is a factor lodading matrix.
//' The item 'mu0' is the mean of the residual.
//' The item 'W' is the variance of the residual.
//' The item 'mu' is the mean of each component.
//' The item 'sigma' is the variance of each component.
//' @export
// [[Rcpp::export]]
Rcpp::List ICMEM( const arma::mat& Y,const arma::ivec& x_int, const arma::sp_mat& Adj, arma::mat& A_int, arma::vec& mu0_int, arma::mat W_int, arma::mat& mu_int, arma::cube& sigma_int, const arma::vec& lambda_grid, const arma::vec& alpha, const arma::vec&  beta_grid, const int& maxIter_ICM, const int& maxIter)	{
  int n = Y.n_rows, p=Y.n_cols, D=A_int.n_cols, K = mu_int.n_cols;
  double diff,diff_old=INFINITY;
  int iter,k,i,m;
  double beta=0.5;
  double lambda=1;
  double tol=1e-6;
  mat gam(n,K);

  ivec x = x_int;	// label
  mat A=A_int;
  vec mu0 = mu0_int;
  mat W=W_int;
  mat mu=mu_int;
  cube sigma = sigma_int;

  mat pxgn(n, K); // p(x_i | x_{N_i})
  pxgn.fill(1.0/K);
  vec pygn = ones<vec>(n);		  // p(y_i | x_{N_i})
  mat pygx = zeros<mat>(n, K);   // p(y_i | x_i = k)

  // Parameter Expansion; double lam = 1.0;
  double ell = 0;
  vec LogLik(maxIter);
  LogLik(0) = INFINITY;

  for(iter=1;iter< maxIter;iter++){

    mat A_new(p,D);
    mat A_denom(D,D);
    mat A_numer(p,D);
    vec mu0_new(p);
    mat W_new(p,p);
    mat mu_new(D,K);
    cube sigma_new(D,D,K);

    // ICM, List;
    // update x and pxgn
    cout<<"ICM"<<endl;
    List fitICM = runICM_sp(Y, A, mu0, W, x, mu, sigma, Adj, alpha, beta, lambda, maxIter_ICM);
    ivec xHat = fitICM["x"];
    x = xHat;
    mat U = fitICM["U"];
    mat Uy = fitICM["Uy"];
    pygx = Uy; // save for latter use, will delete later.
    mat pxgnHat = fitICM["pxgn"];
    pxgn = pxgnHat;
    vec energy = fitICM["energy"];
    LogLik(iter) = min(energy);

    // E-step, update gamma.
    gam = runEstep(U);

    // update beta_grid search.
    int ng_beta = beta_grid.n_elem;
    vec objBetaVec(ng_beta);
    for(k=0; k < ng_beta; ++k){
      objBetaVec(k) = obj_beta(x, gam, Adj, K, alpha, beta_grid(k));
    }

    beta = beta_grid(index_max(objBetaVec));
    mat val_zero(n,K);

    //M-step
    for(i=0;i<n;i++){
      vec y=trans(Y.row(i));
      vec yplus=y.elem(find(abs(y)>=tol));
      uvec nonzero_index=find(abs(y)>=tol);
      uvec zero_index=find(abs(y)<tol);
      List ConditionalExpectation =calConditionalDistribution(y, A,  mu0,  W,  mu, sigma, lambda);
      m=ConditionalExpectation["num_zero"];
      mat mu_exp=ConditionalExpectation["mu"];
      cube sigma_exp=ConditionalExpectation["sigma"];

      arma::mat EZ(D,K);
      arma::cube EZZT(D,D,K);
      arma::cube EZYT(D,p,K);
      arma::mat EY(p,K);
      arma::mat EY2(p,K);
      arma::mat EM(p,K);
      arma::mat EYM(p,K);
      arma::mat EM2(p,K);
      vec temp(p);
      vec temp1(p);

      EZ=mu_exp.rows(0,D-1);

      for(k=0;k<K;k++){
        vec mu_k=mu.col(k);
        mat sigma_k=sigma.slice(k);

        //EZ.col(k)=mu_exp.col(k).subvec(0,D-1);
        EZZT.slice(k)=sigma_exp.slice(k).submat(0,0,D-1,D-1)+EZ.col(k)*trans(EZ.col(k));

        temp.elem(zero_index)=mu_exp.col(k).subvec(D,D+m-1);
        temp.elem(nonzero_index)=yplus;
        EY.col(k)=temp;

        EZYT.slice(k).cols(zero_index)=sigma_exp.slice(k).submat(0,D,D-1,D+m-1)+EZ.col(k)*trans(temp.elem(zero_index));
        EZYT.slice(k).cols(nonzero_index)=EZ.col(k)*trans(yplus);

        vec temp2=diagvec(sigma_exp.slice(k));
        temp1.elem(zero_index)=square(temp.elem(zero_index))+temp2.subvec(D,D+m-1);
        temp1.elem(nonzero_index)=square(temp.elem(nonzero_index));
        EY2.col(k)=temp1;

        EM.col(k)=A*EZ.col(k)+mu0;
        EYM.col(k)=sum(A%trans(EZYT.slice(k)),1)+mu0%EY.col(k);
        EM2.col(k)=diagvec(A*EZZT.slice(k)*trans(A))+2*mu0%(A*EZ.col(k))+mu0%mu0;

        val_zero(i,k)=sum(square(temp.elem(zero_index)));

        A_denom=A_denom+gam(i,k)*EZZT.slice(k);
        A_numer=A_numer+gam(i,k)*trans(EZYT.slice(k)-EZ.col(k)*trans(mu0));

        mu0_new=mu0_new+gam(i,k)*(EY.col(k)-A*EZ.col(k));

        W_new.diag()=W_new.diag()+gam(i,k)*(EY2.col(k)+EM2.col(k)-2*EYM.col(k));
        mu_new.col(k)=mu_new.col(k)+gam(i,k)*EZ.col(k);
        sigma_new.slice(k)=sigma_new.slice(k)+gam(i,k)*(mu.col(k)*trans(mu.col(k))-EZ.col(k)*trans(mu.col(k))-mu.col(k)*trans(EZ.col(k))+EZ.col(k)*trans(EZ.col(k)));
      }
    }

    A_new=A_numer*A_denom.i();
    mu0_new=mu0_new/n;
    W_new=W_new/n;

    for(k=0;k<K;k++){
      mu_new.col(k)=mu_new.col(k)/sum(gam.col(k));
      sigma_new.slice(k)=sigma_new.slice(k)/sum(gam.col(k));
      sigma_new.slice(k) += (1e-6)*eye(D,D);
    }

    int ng_lambda = lambda_grid.n_elem;
    vec objLambdaVec(ng_lambda);
    for(int j=0; j< ng_lambda; j++){
      objLambdaVec(j) = obj_lambda(Y,val_zero,gam,lambda_grid(j));
    }

    lambda = lambda_grid(index_max(objLambdaVec));

    diff=(accu(abs(A-A_new))/(p*D)+accu(abs(mu0-mu0_new))/(p*p)+accu(abs(W-W_new))/p+accu(abs(mu-mu_new))/(D*K)+accu(abs(sigma-sigma_new))/(D*D*K))/(accu(abs(A_new))/(p*D)+accu(abs(mu0_new))/(p*p)+accu(abs(W_new))/p+accu(abs(mu_new))/(D*K)+accu(abs(sigma_new))/(D*D*K));
    if(diff<1e-2||diff>diff_old)
      break;

    diff_old=diff;

    A=A_new;
    mu0=mu0_new;
    W=W_new;
    mu=mu_new;
    sigma=sigma_new;


    // if ( LogLik(iter) - LogLik(iter - 1) > 1e-5 ){
    //   //perror("The energy failed to decrease!");
    //   break;
    // }
    //
    // //if (abs(LogLik(iter) - LogLik(iter - 1)) < 1e-5 || rcond(sigma) < 1e-7) {
    // if (abs(LogLik(iter) - LogLik(iter - 1)) < 1e-5) {
    //   cout << "Converged at Iteration = " << iter  << endl;
    //   break;
    // }
  }

  List output = List::create(
    Rcpp::Named("x")=x,
    Rcpp::Named("beta")=beta,
    Rcpp::Named("gam") = gam,
    Rcpp::Named("mu0") = mu0,
    Rcpp::Named("W") = W,
    Rcpp::Named("lambda")=lambda,
    Rcpp::Named("A") = A,
    Rcpp::Named("mu") = mu,
    Rcpp::Named("sigma") = sigma);

  return output;
}

