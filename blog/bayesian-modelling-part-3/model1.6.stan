// model1.6.stan
data{
    //// Passing Data Dimensions/Sizes
    int<lower=10>   num_samples;
    int<lower=1>    num_customers; // is same as num_samples
    int<lower=1>    num_os; //number of types of operating system
    int<lower=1>    num_devices; //number of types of devices
    int<lower=1>    num_software_versions; //number of types of software versions
    int<lower=1>    num_countries; // number of countries

    //// Passing Data elements to model
    int<lower=1, upper=num_os>                os[num_samples]; // operating system column
    int<lower=1, upper=num_devices>           device_type[num_samples]; // device_type column
    int<lower=1, upper=num_software_versions> software_version[num_samples]; // software_version column
    int<lower=1, upper=num_countries>         country[num_samples]; // country column

    //// Hierarchy data
    // Children in the tree hierarchy
    int<lower=1, upper=num_devices>           os_children[num_os]; // l0(os) children->l1(device)
    int<lower=1, upper=num_software_versions> device_children[num_devices]; // l1(device) children->l2(sw_version)
    int<lower=1, upper=num_customers>         sw_version_children[num_software_versions]; // l2(sw) children->customers

    // Parent
    int<lower=1, upper=num_os>        device_parent[num_devices];
    int<lower=1, upper=num_devices>   sw_version_parent[num_software_versions];

    //// Passing Target Revenue
    vector<lower=0>[num_samples] revenue;  // revenue 
    vector<lower=0>[num_samples] discount;
    int<lower=0, upper=1> install[num_samples]; // whether game was installed or not

    //Passing Priors
    real LOCATION_mu_root;
    real<lower=0> SCALE_mu_root;
    real<lower=0> SCALE_sd_root;                 // scale of between groups variation at root
    real<lower=0> SCALE_sigma_root;              // scale of within groups variation at root
    real<lower=0.05> SCALE_beta_country;
}

transformed data{
    int length_mu_raw_params; // length of raw mu params
    int length_sigma_raw_params; // length of raw sigma params
    int length_sd_raw_params; // length of raw sd params

    // Hyper-prior widths. These are on the LOG scale, i.e. sigma_root has a
    // lognormal(log(SCALE_sigma_root), PRIOR_LOG_SCALE_ROOT) prior, so a value
    // of 0.7 means "the root scale is within a factor of ~4 of SCALE_sigma_root
    // with 95% probability".
    real PRIOR_LOG_SCALE_ROOT;   // width of the lognormal prior on the root scales
    real PRIOR_SCALE_TAU;        // half-normal scale on the level-to-level spread
    real<lower=0> MIN_SIGMA_DISC; // floor on the discount residual sd 

    PRIOR_LOG_SCALE_ROOT = 0.7;
    PRIOR_SCALE_TAU      = 0.5;

    MIN_SIGMA_DISC = 0.02 * sd(discount);

    length_mu_raw_params    = 1;
    length_sigma_raw_params = 1;
    length_sd_raw_params    = 1;
    // Covers first layer of the hierarchy i.e @os level
    if(num_os>1){
        length_mu_raw_params    = length_mu_raw_params + num_os;
        length_sigma_raw_params = length_sigma_raw_params + num_os;
        length_sd_raw_params    = length_sd_raw_params + num_os;
    }
    // Covers second layer of the hierarchy i.e @device_type level
    // for each os - ios /android, if children>1, each child i.e device will have a mu/sigma/sd param
    for(n in 1:num_os){
        if(os_children[n]>1){
            length_mu_raw_params    = length_mu_raw_params + os_children[n];
            length_sigma_raw_params = length_sigma_raw_params + os_children[n];
            length_sd_raw_params    = length_sd_raw_params + os_children[n];
        }
    }
    // Covers third layer of the hierarchy i.e @software_version level
    // for each software version, if the parent device has multiple software versions and if software version has multiple children
    for(n in 1:num_software_versions){
        int parent;
        parent = sw_version_parent[n];
        if(sw_version_children[n]>1 && device_children[parent]>1){
            length_mu_raw_params    = length_mu_raw_params + 1;
            length_sigma_raw_params = length_sigma_raw_params + 1;
        }
    }
}

parameters{
    // Hierarchical relationship params. ALL of these are now unconstrained
    // standard-normal z-scores 
    vector[length_mu_raw_params]    mu_raw;
    vector[length_sigma_raw_params] sigma_raw; // z-scores for log(sigma)
    vector[length_sd_raw_params]    sd_raw;    // z-scores for log(sd)

    // How much a scale is allowed to move from one level of the tree to the
    // next. tau -> 0 means "all groups share the parent's scale"; the data
    // decide how much heterogeneity there is instead of a hard [1, 1.25] wall.
    real<lower=0> tau_sigma;
    real<lower=0> tau_sd;

    // Discount regression
    vector<lower=0>[num_countries] beta_country;
    real<lower=MIN_SIGMA_DISC> sigma_disc_error;
}

transformed parameters{
    /// Root params
    real mu_root;
    real<lower=0> sigma_root;
    real<lower=0> sd_root;

    vector[num_os] mu_os;
    vector<lower=0>[num_os] sd_os; // variation among devicew in an os
    vector<lower=0>[num_os] sigma_os;

    vector[num_devices] mu_device;
    vector<lower=0>[num_devices] sd_device;
    vector<lower=0>[num_devices] sigma_device;

    vector[num_software_versions] mu_sw_version;
    vector<lower=0>[num_software_versions] sigma_sw_version;

    vector[num_customers] revenue_hierarchy; // expected revenue for each customer
    vector[num_customers] sigma_revenue;     // sd of revenue for each customer

    {
        int ctr_mu; // counter for mu
        int ctr_sd; // counter for sd
        int parent_idx; // parent's index in the hierarchy
        ctr_mu = 1;
        ctr_sd = 1;

        /// Root level of tree
        mu_root    = LOCATION_mu_root + SCALE_mu_root * mu_raw[ctr_mu];
        // lognormal, centred on the scale the user supplies, but free to move
        // an order of magnitude in either direction.
        sigma_root = SCALE_sigma_root * exp(PRIOR_LOG_SCALE_ROOT * sigma_raw[ctr_mu]);
        sd_root    = SCALE_sd_root    * exp(PRIOR_LOG_SCALE_ROOT * sd_raw[ctr_sd]);
        ctr_mu = ctr_mu + 1;
        ctr_sd = ctr_sd + 1;

        /// First Layer of Hierarchy - operating system
        for( i in 1:num_os ){
            if(num_os>1){
                mu_os[i]    = mu_root + sd_root * mu_raw[ctr_mu];
                // multiplicative, but exp(tau*z) can be < 1 as well as > 1
                sigma_os[i] = sigma_root * exp(tau_sigma * sigma_raw[ctr_mu]);
                sd_os[i]    = sd_root    * exp(tau_sd    * sd_raw[ctr_sd]);
                ctr_mu = ctr_mu + 1;
                ctr_sd = ctr_sd + 1;
            }else{
                mu_os[i]    = mu_root;
                sigma_os[i] = sigma_root;
                sd_os[i]    = sd_root;
            }
        }
        /// Second Layer of Hierarchy - devices
        for( i in 1:num_devices){
            parent_idx = device_parent[i]; // the parent of device in the hierarchy is the operating system
            if(os_children[parent_idx]>1){
                mu_device[i]    = mu_os[parent_idx] + sd_os[parent_idx] * mu_raw[ctr_mu];
                sigma_device[i] = sigma_os[parent_idx] * exp(tau_sigma * sigma_raw[ctr_mu]);
                sd_device[i]    = sd_os[parent_idx]    * exp(tau_sd    * sd_raw[ctr_sd]);
                ctr_mu = ctr_mu + 1;
                ctr_sd = ctr_sd + 1;
            }else{
                mu_device[i]    = mu_os[parent_idx];
                sigma_device[i] = sigma_os[parent_idx];
                sd_device[i]    = sd_os[parent_idx];
            }
        }
        /// Third Layer of Hierarchy - software versions
        for( i in 1:num_software_versions){
            parent_idx = sw_version_parent[i]; // the parent of software versions in the hierarchy is devices
            if(device_children[parent_idx]>1 && sw_version_children[i]>1 ){
                mu_sw_version[i]    = mu_device[parent_idx] + sd_device[parent_idx] * mu_raw[ctr_mu];
                sigma_sw_version[i] = sigma_device[parent_idx] * exp(tau_sigma * sigma_raw[ctr_mu]);
                ctr_mu = ctr_mu + 1;
            }else{
                mu_sw_version[i]    = mu_device[parent_idx];
                sigma_sw_version[i] = sigma_device[parent_idx];
            }
        }
    }

    // NOTE: revenue_hierarchy is now the EXPECTED revenue of the group a
    // customer belongs to, not a free per-customer latent draw
    revenue_hierarchy = mu_sw_version[software_version];
    sigma_revenue     = sigma_sw_version[software_version];
}

model{
    // PRIORS
    mu_raw    ~ normal(0,1);
    sigma_raw ~ normal(0,1);   // => lognormal priors on sigma_root/os/device/sw
    sd_raw    ~ normal(0,1);   // => lognormal priors on sd_root/os/device

    tau_sigma ~ normal(0, PRIOR_SCALE_TAU); // half-normal (lower=0 in decl.)
    tau_sd    ~ normal(0, PRIOR_SCALE_TAU);

    beta_country ~ lognormal(log(0.25 * SCALE_beta_country), 0.5);

    // Half-normal on the excess above the floor. With noiseless discount data
    // the posterior simply piles up at MIN_SIGMA_DISC, which is a clean
    // exponential tail in the unconstrained space rather than a funnel.
    sigma_disc_error ~ normal(MIN_SIGMA_DISC, 0.5 * sd(discount));

    // LIKELIHOOD
    target += normal_lpdf( revenue  | revenue_hierarchy, sigma_revenue );
    target += normal_lpdf( discount | beta_country[country] .* revenue, sigma_disc_error );
}

generated quantities{
    vector[num_samples] log_lik;      // for loo / waic in the evaluation notebook
    vector[num_samples] revenue_rep;  // posterior predictive draws

    for(n in 1:num_samples){
        log_lik[n] = normal_lpdf(revenue[n]  | revenue_hierarchy[n], sigma_revenue[n])
                   + normal_lpdf(discount[n] | beta_country[country[n]] * revenue[n], sigma_disc_error);
        revenue_rep[n] = normal_rng(revenue_hierarchy[n], sigma_revenue[n]);
    }
}
