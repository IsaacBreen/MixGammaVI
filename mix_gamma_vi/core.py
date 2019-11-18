import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import scipy

tfd = tfp.distributions
lgamma = tf.math.lgamma
polygamma = tf.math.polygamma
log = tf.math.log
exp = tf.math.exp
reduce_sum = tf.math.reduce_sum
reduce_mean = tf.math.reduce_mean

dtype = tf.float64
intdtype = tf.int32
npdtype = np.float64

pi_numeric = tf.cast(np.pi, dtype)
e_numeric = tf.cast(np.e, dtype)

@tf.function
def take_after(a, i, n):
    if i>a.shape[0] or i<0:
        i = i%a.shape[0]
    if i+n>a.shape[0]:
        return tf.concat([a[i:], a[:(n-(a.shape[0]-i))]], axis=0)
    else:
        return a[i:i+n]

    
@tf.function
def to_dtype(x):
    return tf.cast(x, dtype)


@tf.function
def _mix_gamma_vi_1(x, K=1, w0=10000., wT=1., r=1e-10, s=1e-10, c=1e-10, d=1e-10, eps=1, BATCH_SIZE=250, MAX_ITERATIONS=10000, 
        MIN_ITERATIONS=0, BATCH_SIZE_MULTIPLIER=100, MIN_WARMUP=0, MAX_WARMUP=10000, TOLERANCE=1/10000, ELBO_TICK=5, RUNNING_ELBO_SIZE=10, 
        AHAT_STEPS=2, VERBOSE=False, RETURN_HISTORY=False, RETURN_DTYPE=tf.float64):

    # If return datatype is not specified, let it be the same as the input datatype for x
    if RETURN_DTYPE is None:
        RETURN_DTYPE = x.dtype

    # Convert arguments to TensorFlow objects
    N = x.shape[0]
    x,w0,wT,r,s,c,d,eps = [tf.cast(var, dtype) for var in [x,w0,wT,r,s,c,d,eps]]
    K_float = to_dtype(K)
    x = tf.reshape(x, (-1,1))

    # Calculate the prior strength discount factor k
    w = w0
    k=(w0/wT)**(1/MAX_ITERATIONS)

    # Set initial values
    elbo = tf.constant(0, dtype)

    x_mean = tf.math.reduce_mean(x)
    x_var = tf.math.reduce_mean( (x - x_mean)**2 )

    start_means = tf.reshape(tf.linspace(tf.maximum(-1.5*x_var**0.5 + x_mean, 1e-3), 1.5*x_var**0.5 + x_mean, K) , (1,K))
    start_vars = tf.fill((1,K), x_var/K_float**2)
    
    zeta = tf.cast(tf.fill((1,K), N/K_float), dtype=dtype) + w
    gamma = start_means*10000
    lambda_ = start_vars*10000
    
    ahat = start_means**2/start_vars
    sigma_sq = 1/(polygamma(to_dtype(1), ahat)*(s + N/K))
    
    i = tf.constant(0, intdtype)

    # Setup data-structures to store values if RETURN_HISTORY is True
    if RETURN_HISTORY:
        zeta_history     = tf.zeros((MAX_ITERATIONS,K), dtype)
        ahat_history     = tf.zeros((MAX_ITERATIONS,K), dtype)
        sigma_sq_history = tf.zeros((MAX_ITERATIONS,K), dtype)
        gamma_history    = tf.zeros((MAX_ITERATIONS,K), dtype)
        lambda_history   = tf.zeros((MAX_ITERATIONS,K), dtype)
        elbo_history     = tf.zeros(MAX_ITERATIONS, dtype)
    else:
        zeta_history     = tf.constant(0)
        ahat_history     = tf.constant(0)
        sigma_sq_history = tf.constant(0)
        gamma_history    = tf.constant(0)
        lambda_history   = tf.constant(0)
        elbo_history     = tf.constant(0)


    running_elbo = tf.zeros(RUNNING_ELBO_SIZE*2)

    x_shuffled = tf.random.shuffle(x)
    logx = log(x)
    logx_shuffled = log(x_shuffled)

    j = tf.constant(0, intdtype)
    
    # Some counters and a flag
    BREAK_COUNTER = tf.constant(0)
    ELBO_COUNTER = tf.constant(0)
    CAVI_PHASE = False

    # Begin variational inference
    for i in tf.range(start=0, limit=MAX_ITERATIONS-1):

        # Discount the prior strength
        w = w/k

        # Resample the data
        if (j+1)*BATCH_SIZE>x.shape[0]:
            x_shuffled = tf.random.shuffle(x)
            logx_shuffled = log(x_shuffled)
            j = tf.constant(0)
        xb = x_shuffled[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        logxb  = logx_shuffled[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        j += 1
            
        # Compute q_{i,j}
        q = polygamma(to_dtype(0), zeta) - polygamma(to_dtype(0), K_float*w+N) + ahat*(polygamma(to_dtype(0), gamma) - log(lambda_)) \
            - lgamma(ahat) - 1/2*sigma_sq*polygamma(to_dtype(1), ahat) + (ahat-1)*logxb - gamma/lambda_*xb
        q = q - tf.reshape(tf.math.reduce_max(q, 1), (-1,1))
        q = tf.math.exp(q)
        q = q/tf.reshape(tf.reduce_sum(q, -1), (-1,1))
                
        
        # Calculate and store some often-used values
        batchsize_correction = tf.cast(N/BATCH_SIZE, dtype)
        q_summed_over_data = batchsize_correction*tf.reshape(tf.reduce_sum(q, 0), (1,-1))
        q_times_x_summed_over_data = batchsize_correction*tf.reshape(tf.reduce_sum(q*xb, 0), (1,-1))
        q_times_log_x_summed_over_data = batchsize_correction*tf.reshape(tf.reduce_sum(q*logxb, 0), (1,-1))

        # Variational updates for zeta, gamma (here called gamma) and lambda (here called lambda 2)
        zeta = (1-eps)*zeta + eps*(w + q_summed_over_data)
        gamma = (1-eps)*gamma + eps*(c + ahat*q_summed_over_data)
        lambda_ = (1-eps)*lambda_ + eps*(d + q_times_x_summed_over_data)
        sigma_sq = (1-eps)*sigma_sq + eps*1/(polygamma(to_dtype(1), ahat)*(s + N/K))

        # Newton-Raphson algorithm for the variational distribution of alpha
        ahat_nr = ahat
        for _ in tf.range(AHAT_STEPS):
            ahat_nr = ahat_nr + ( (polygamma(to_dtype(0),gamma) - log(lambda_))*q_summed_over_data + q_times_log_x_summed_over_data + r \
                    - (q_summed_over_data + s)*(polygamma(to_dtype(0), ahat_nr) + 1/2*sigma_sq * polygamma(to_dtype(2), ahat_nr))) \
                /( (q_summed_over_data+s)*(polygamma(to_dtype(1), ahat_nr) + 1/2*sigma_sq*polygamma(to_dtype(3), ahat_nr)) )
            ahat_nr = tf.abs(ahat_nr)
            ahat_nr = tf.clip_by_value(ahat_nr, clip_value_min=1e-30, clip_value_max=1e+30)

        ahat = (1-eps)*ahat + eps*ahat_nr
    
    
        # Store values
        if RETURN_HISTORY:
            zeta_history = tf.tensor_scatter_nd_update(zeta_history, [[i]], zeta)
            ahat_history = tf.tensor_scatter_nd_update(ahat_history, [[i]], ahat)
            sigma_sq_history = tf.tensor_scatter_nd_update(sigma_sq_history, [[i]], sigma_sq)
            gamma_history = tf.tensor_scatter_nd_update(gamma_history, [[i]], gamma)
            lambda_history = tf.tensor_scatter_nd_update(lambda_history, [[i]], lambda_)

            
        # Calculate the ELBO every ELBO_TICK iterations
        if i%ELBO_TICK==0:

            elbo_constants = -tf.cast(K_float*lgamma(w) - lgamma(w*K_float), dtype=dtype) 

            E_joint_log_prob = elbo_constants + reduce_sum((w+q_summed_over_data-1)*(polygamma(to_dtype(0), zeta) \
                - tf.cast(polygamma(to_dtype(0), K_float*w+N), dtype=dtype)) + c*tf.cast(log(d), dtype=dtype) \
                - lgamma(c) + (c-1 + ahat*q_summed_over_data)*( polygamma(to_dtype(0), gamma) - log(lambda_) ) \
                - d*gamma/lambda_ + r*ahat - (lgamma(ahat) + 1/2*sigma_sq*polygamma(to_dtype(1), ahat))*(s+q_summed_over_data) \
                + (ahat-1)*q_times_log_x_summed_over_data - gamma/lambda_*q_times_x_summed_over_data)

            entropy = reduce_sum((gamma - log(lambda_) + lgamma(gamma)) + (1-gamma)*polygamma(to_dtype(0), gamma) \
                + 1/2*log(2*pi_numeric*e_numeric*sigma_sq) + lgamma(zeta) \
                + zeta*polygamma(to_dtype(0), reduce_sum(zeta)) - (zeta-1)*polygamma(to_dtype(0), zeta)) \
                - lgamma(reduce_sum(zeta)) - K_float*polygamma(to_dtype(0), reduce_sum(zeta))

            elbo = E_joint_log_prob + entropy            

            # Store ELBO
            if RETURN_HISTORY:
                elbo_history = tf.tensor_scatter_nd_update(elbo_history, [[ELBO_COUNTER]], [elbo])

            # Update an ELBO matrix to calculate a running mean of the ELBO
            running_elbo = tf.tensor_scatter_nd_update(running_elbo, [[ELBO_COUNTER%(RUNNING_ELBO_SIZE*2)]], [elbo])

            if BREAK_COUNTER>RUNNING_ELBO_SIZE*2 and i>MIN_WARMUP or i>MAX_WARMUP:
                # Calculate the graident of the running mean of the elbo
                elbo_mean1 = reduce_mean(take_after(running_elbo, ELBO_COUNTER%(RUNNING_ELBO_SIZE*2), RUNNING_ELBO_SIZE))
                elbo_mean2 = reduce_mean(take_after(running_elbo, ELBO_COUNTER%(RUNNING_ELBO_SIZE*2)+RUNNING_ELBO_SIZE, RUNNING_ELBO_SIZE))
                gradient = (elbo_mean1 - elbo_mean2)/RUNNING_ELBO_SIZE/elbo_mean1

                if gradient<TOLERANCE:
                    if BATCH_SIZE < N or not CAVI_PHASE:
                        # Increase batch size
                        CAVI_PHASE = True
                        BATCH_SIZE = tf.minimum(BATCH_SIZE*BATCH_SIZE_MULTIPLIER, N)
                        BREAK_COUNTER = tf.cast(tf.constant(RUNNING_ELBO_SIZE/2), tf.int32)
                        eps=tf.constant(1., dtype)
                        w = tf.constant(1., dtype)
                        k = tf.constant(1., dtype)
                    elif i>MIN_ITERATIONS:
                        # Done
                        break
                        
            ELBO_COUNTER = ELBO_COUNTER + 1
            BREAK_COUNTER = BREAK_COUNTER + 1

    if RETURN_HISTORY:
        return [tf.cast(var[:i], RETURN_DTYPE) for var in \
                [zeta_history, ahat_history, sigma_sq_history, gamma_history, lambda_history]] + [elbo_history[:ELBO_COUNTER],]
    else:
        return [tf.cast(var, RETURN_DTYPE) for var in [zeta, ahat, sigma_sq, gamma, lambda_, elbo]]


@tf.function
def _mix_gamma_vi_2(x, K=1, w0=10000., wT=1., r=1e-10, s=1e-10, xi=1e-10, tau=1e-10, eps=1, BATCH_SIZE=250, MAX_ITERATIONS=10000, 
        MIN_ITERATIONS=0, BATCH_SIZE_MULTIPLIER=100, MIN_WARMUP=0, MAX_WARMUP=10000, TOLERANCE=1/10000, ELBO_TICK=5, RUNNING_ELBO_SIZE=10, 
        AHAT_STEPS=2, VERBOSE=False, RETURN_HISTORY=False, RETURN_DTYPE=None):

    # If return datatype is not specified, let it be the same as the input datatype for x
    if RETURN_DTYPE is None:
        RETURN_DTYPE = x.dtype

    # Convert arguments to TensorFlow objects
    N = x.shape[0]
    x,w0,wT,r,s,xi,tau,eps = [tf.cast(var, dtype) for var in [x,w0,wT,r,s,xi,tau,eps]]
    K_float = to_dtype(K)
    x = tf.reshape(x, (-1,1))

    # Calculate the prior strength discount factor k
    w = w0
    k=(w0/wT)**(1/MAX_ITERATIONS)

    # Set initial values
    elbo = tf.constant(0, dtype)

    x_mean = tf.math.reduce_mean(x)
    x_var = tf.math.reduce_mean( (x - x_mean)**2 )

    start_means = tf.reshape(tf.linspace(tf.maximum(-1.5*x_var**0.5 + x_mean, 1e-3), 1.5*x_var**0.5 + x_mean, K) , (1,K))
    start_vars = tf.fill((1,K), x_var/K_float**2)

    zeta = tf.cast(tf.fill((1,K), N/K_float), dtype=dtype) + w
    gamma = tf.cast(tf.fill((1,K), 1.), dtype=dtype)*10000
    lambda_ = start_means*10000

    ahat = start_means**2/start_vars
    sigma_sq = tf.cast(tf.fill((1,K), 1e-5), dtype)
    
    i = tf.constant(0, intdtype)

    # Setup data-structures to store values if RETURN_HISTORY is True, otherwise set dummy values
    if RETURN_HISTORY:
        zeta_history     = tf.zeros((MAX_ITERATIONS,K), dtype)
        ahat_history     = tf.zeros((MAX_ITERATIONS,K), dtype)
        sigma_sq_history = tf.zeros((MAX_ITERATIONS,K), dtype)
        gamma_history    = tf.zeros((MAX_ITERATIONS,K), dtype)
        lambda_history   = tf.zeros((MAX_ITERATIONS,K), dtype)
        elbo_history     = tf.zeros(MAX_ITERATIONS, dtype)
    else:
        zeta_history     = tf.constant(0)
        ahat_history     = tf.constant(0)
        sigma_sq_history = tf.constant(0)
        gamma_history    = tf.constant(0)
        lambda_history   = tf.constant(0)
        elbo_history     = tf.constant(0)


    running_elbo = tf.zeros(RUNNING_ELBO_SIZE*2)
    
    x_shuffled = tf.random.shuffle(x)
    logx = log(x)
    logx_shuffled = log(x_shuffled)
    
    j = tf.constant(0, intdtype)

    # Some counters and a flag
    BREAK_COUNTER = tf.constant(0)
    ELBO_COUNTER = tf.constant(0)
    CAVI_PHASE = False
    
    # Begin variational inference
    for i in tf.range(start=0, limit=MAX_ITERATIONS-1):

        # Discount the prior strength
        w = w/k

        # Resample the data
        if (j+1)*BATCH_SIZE>x.shape[0]:
            x_shuffled = tf.random.shuffle(x)
            logx_shuffled = log(x_shuffled)
            j = tf.constant(0)
        xb = x_shuffled[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        logxb  = logx_shuffled[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        j += 1

        # Compute q_{i,j}
        q = polygamma(to_dtype(0), zeta) - polygamma(to_dtype(0), K_float*w+N) + 1/2*log(ahat) - sigma_sq/(4*ahat**2) \
             + ahat*(polygamma(to_dtype(0), gamma) - log(lambda_) + 1) + (ahat-1)*logxb - ahat*(gamma/lambda_)*xb
        q = q - tf.reshape(tf.math.reduce_max(q, 1), (-1,1))
        q = tf.math.exp(q)
#         print(q.shape)
        q = q/tf.reshape(tf.reduce_sum(q, -1), (-1,1))

        # Calculate and store some often-used values
        batchsize_correction = tf.cast(N/BATCH_SIZE, dtype)
        q_summed_over_data = batchsize_correction*tf.reshape(tf.reduce_sum(q, 0), (1,-1))
        q_times_x_summed_over_data = batchsize_correction*tf.reshape(tf.reduce_sum(q*xb, 0), (1,-1))
        q_times_log_x_summed_over_data = batchsize_correction*tf.reshape(tf.reduce_sum(q*logxb, 0), (1,-1))
                
        # Variational updates for zeta, gamma (here called gamma), lambda (here called lambda 2), and sigma_sq
        zeta = (1-eps)*zeta + eps*(w + q_summed_over_data)
        gamma = (1-eps)*gamma + eps*(xi + ahat*q_summed_over_data)
        lambda_ = (1-eps)*lambda_ + eps*(tau + ahat*q_times_x_summed_over_data)
        sigma_sq = (1-eps)*sigma_sq + eps*1/( s*polygamma(to_dtype(1), ahat) + 1/(2*ahat**2)*q_summed_over_data)

        # Newton-Raphson algorithm for the variational distribution of alpha
        ahat_nr = ahat
        for _ in tf.range(AHAT_STEPS):
            polygamma_0_ahat = polygamma(to_dtype(0), ahat_nr)
            polygamma_1_ahat = polygamma(to_dtype(1), ahat_nr)
            polygamma_2_ahat = polygamma(to_dtype(2), ahat_nr)
            polygamma_3_ahat = polygamma(to_dtype(3), ahat_nr)
            
            ahat_nr = ahat_nr - ( (polygamma(to_dtype(0), gamma) - log(lambda_))*q_summed_over_data \
                    + r + q_times_log_x_summed_over_data - q_times_x_summed_over_data*gamma/lambda_ \
                    + ( - sigma_sq/(2*ahat_nr**2) + 1 + log(ahat_nr) - polygamma_0_ahat \
                    - 1/2*sigma_sq*polygamma_2_ahat)*q_summed_over_data 
                    - s*(polygamma_0_ahat + 1/2*sigma_sq*polygamma_2_ahat)) \
                /( ( 1/ahat_nr - polygamma_1_ahat - 1/2*sigma_sq*polygamma_3_ahat)*q_summed_over_data \
                    - s*(polygamma_1_ahat + 1/2*sigma_sq*polygamma_3_ahat))
            ahat_nr = tf.abs(ahat_nr)

        ahat = (1-eps)*ahat + eps*ahat_nr
        
        # Store values
        if RETURN_HISTORY:
            zeta_history = tf.tensor_scatter_nd_update(zeta_history, [[i]], zeta)
            ahat_history = tf.tensor_scatter_nd_update(ahat_history, [[i]], ahat)
            sigma_sq_history = tf.tensor_scatter_nd_update(sigma_sq_history, [[i]], sigma_sq)
            gamma_history = tf.tensor_scatter_nd_update(gamma_history, [[i]], gamma)
            lambda_history = tf.tensor_scatter_nd_update(lambda_history, [[i]], lambda_)


        # Calculate the ELBO every ELBO_TICK iterations
        if i%ELBO_TICK==0:
            
            elbo_constants = -tf.cast(K_float*lgamma(w) - lgamma(w*K_float), dtype=dtype) 

            E_joint_log_prob = elbo_constants + reduce_sum((w+q_summed_over_data-1)*(polygamma(to_dtype(0), zeta) \
                - tf.cast(polygamma(to_dtype(0), K_float*w+N), dtype=dtype)) + xi*tf.cast(log(tau), dtype=dtype) - lgamma(xi) \
                + (1 - xi - ahat*q_summed_over_data)*(log(lambda_) - polygamma(to_dtype(0), gamma)) \
                - tau*gamma/lambda_ + (-1/2*log(2*pi_numeric) + 1/2*log(ahat) - sigma_sq/(4*ahat**2) + ahat)*q_summed_over_data \
                + (ahat-1)*q_times_log_x_summed_over_data - ahat*gamma/lambda_*q_times_x_summed_over_data)

            entropy = reduce_sum((gamma + log(lambda_) + lgamma(gamma)) - (gamma+1)*polygamma(to_dtype(0), gamma) \
                + 1/2*log(2*pi_numeric*e_numeric*sigma_sq) + lgamma(zeta) \
                + zeta*polygamma(to_dtype(0), reduce_sum(zeta)) - (zeta-1)*polygamma(to_dtype(0), zeta)) \
                - lgamma(reduce_sum(zeta)) - K_float*polygamma(to_dtype(0), reduce_sum(zeta))
            
            elbo = E_joint_log_prob + entropy

            # Store ELBO
            if RETURN_HISTORY:
                elbo_history = tf.tensor_scatter_nd_update(elbo_history, [[ELBO_COUNTER]], [elbo])
                
            # Update an ELBO matrix to calculate a running mean of the ELBO
            running_elbo = tf.tensor_scatter_nd_update(running_elbo, [[ELBO_COUNTER%(RUNNING_ELBO_SIZE*2)]], [elbo])

            if BREAK_COUNTER>RUNNING_ELBO_SIZE*2 and i>MIN_WARMUP or i>MAX_WARMUP:
                # Calculate the graident of the running mean of the elbo
                elbo_mean1 = reduce_mean(take_after(running_elbo, ELBO_COUNTER%(RUNNING_ELBO_SIZE*2), RUNNING_ELBO_SIZE))
                elbo_mean2 = reduce_mean(take_after(running_elbo, ELBO_COUNTER%(RUNNING_ELBO_SIZE*2)+RUNNING_ELBO_SIZE, RUNNING_ELBO_SIZE))
                gradient = (elbo_mean1 - elbo_mean2)/RUNNING_ELBO_SIZE/elbo_mean1

                if gradient<TOLERANCE:
                    if BATCH_SIZE < N or not CAVI_PHASE:
                        # Increase batch size
                        CAVI_PHASE = True
                        BATCH_SIZE = tf.minimum(BATCH_SIZE*BATCH_SIZE_MULTIPLIER, N)
                        BREAK_COUNTER = tf.cast(tf.constant(RUNNING_ELBO_SIZE/2), tf.int32)
                        eps=tf.constant(1., dtype)
                        w = tf.constant(1., dtype)
                        k = tf.constant(1., dtype)
                    elif i>MIN_ITERATIONS:
                        # Done
                        break

            ELBO_COUNTER = ELBO_COUNTER + 1
            BREAK_COUNTER = BREAK_COUNTER + 1

    if RETURN_HISTORY:
        return [tf.cast(var[:i], RETURN_DTYPE) for var in \
                [zeta_history, ahat_history, sigma_sq_history, gamma_history, lambda_history]] + [elbo_history[:ELBO_COUNTER],]
    else:
        return [tf.cast(var, RETURN_DTYPE) for var in [zeta, ahat, sigma_sq, gamma, lambda_, elbo]]


parameter_names = ["zeta", "ahat", "sigma_sq", "gamma", "lambda_", "elbo"]

class mix_gamma_vi:

    def __init__(self, x, K=1, parameterisation="mean-shape", **kwargs):
        self.parameterisation = parameterisation
        if parameterisation=="mean-shape":
            self.parameters = _mix_gamma_vi_2(x, K=K, **kwargs)
        elif parameterisation=="shape-rate":
            self.parameters = _mix_gamma_vi_1(x, K=K, **kwargs)
        else:
            stop("parameterisation parameter not recognized. Choose parameterisation=\"mean-shape\" (recommended) or parameterisation=\"shape-rate\"")

    def parameter_dict(self):
        return dict(zip(parameter_names, self.parameters))

    def distribution(self):
        zeta, ahat, sigma_sq, gamma, lambda_, elbo = self.parameters
        if self.parameterisation=="mean-shape":
            dist = tfd.JointDistributionNamed(dict(
                pi    = tfd.Dirichlet(zeta),
                alpha = tfd.Normal(ahat, sigma_sq**0.5),
                mu    = tfd.InverseGamma(gamma, lambda_)))
        else:
            dist = tfd.JointDistributionNamed(dict(
                pi    = tfd.Dirichlet(zeta),
                alpha = tfd.Normal(ahat, sigma_sq**0.5),
                beta  = tfd.InverseGamma(gamma, lambda_)))
        return dist






