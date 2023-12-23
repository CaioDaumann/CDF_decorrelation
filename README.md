# CDF_decorrelation

This repo stores the code that performs the decorrelation betwenn two variables using the Smirnov transformation or also sometimes called quantile morphing.

How can we apply these transformations to perform decorrelation? For example, in this case we have two variables, $\sigma_{m}/m$ and $m_{\gamma\gamma}$,that are correlated, and we want $\sigma_{m}/m$ to be uncorrelated with the mass. 

For that we can separate the mass variables in several bins, in this specific situation from 100 to 180, and the bins are 0.5 wide. now, the events inside the bins will have a $\sigma_{m}/m$ distribution, a diferent one for each bin, given that the variables are correlated.

For each of these bins, we can calculate the CDF of the $\sigma_{m}/m$, and choose one bin with a reference CDF. Then we can use the CDF and morph each bin into the reference CDF. In doing that, every mass bin will have the same $\sigma_{m}/m$ distribution, and thus, the variables will not be correlated anymore.

Disclaimer: This code was not developed by me, I just adapted it to made it run on .parquet files. Also worked on more plotting script, this repo objective is just to keep the code.

More details in: https://www.research-collection.ethz.ch/handle/20.500.11850/579650
