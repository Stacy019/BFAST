# BFAST
Jointly performing dimension reduction and spatial clustering with Bayesian Factor Analysis for zero-inflated Spatial Transcriptomics data.

# Installation
To install the packages "BFAST", firstly, install the "devtools" package. Besides, "BFAST" depends on the "Rcpp" and "RcppArmadillo" package, which also requires appropriate setting of Rtools and Xcode for Windows and Mac OS/X, respectively.

install.packages("devtools")

library(devtools)

install_github("Stacy019/BFAST")

#Usage
Take the STARmap data as example, 
load("starmap_mpfc.RData")
data=starmap_cnts[[1]]
Data=CreateSeuratObject(data,min.cells=3)
Data=NormalizeData(Data)
Data=SCTransform(Data)

spot_x=starmap_info[[1]]$x
spot_y=starmap_info[[1]]$y
Adj=getneighborhood_fast(cbind(spot_x,spot_y),300)

y=t(as.matrix(Data@assays$SCT@data))
y=scale(y,center=F)
out=InitalPara(y,K,q,int.model = "EEE")
res=BFAST::ICMEM(y,out$x_int,Adj,out$A_int,out$mu0_int,diag(as.vector(out$W_int)),out$mu_int,out$sigma_int,lambda_grid=seq(0,2,0.2),alpha=rep(0,K),beta_grid=seq(0,4,0.2), maxIter_ICM=10, maxIter=50)
