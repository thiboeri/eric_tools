import numpy
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
MRG = RNG_MRG.MRG_RandomStreams(1)

def binary_xce(p, t):
    return -(t * numpy.log(p) + (1-t) * numpy.log(1-p))

def cast32(x):
    return numpy.cast['float32'](x)

def trunc(x, n=8):
    return str(x)[:n]

def logit(p):
    return numpy.log(p / (1 - p) )

def binarize(x, threshold=0.5):
    return cast32(x >= threshold)

def sigmoid(x):
    return cast32(1. / (1 + numpy.exp(-x)))

def rectifier(x):
    return T.maximum(x, theano.shared(cast32(0)))

def get_shared_weights(n_in, n_out, interval=None, name='W'):
    if interval==None:
        interval    =   cast32(1./((n_in + n_out)**0.5))
    val = numpy.random.uniform(-interval, interval, size=(n_in, n_out))
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def get_sparse_shared_weights(n_in, n_out, n_nonzeros = 15, sigma=None, name='W'):
    # Init weights with (15) non-zero entries per unit, gaussian
    if sigma==None:
        sigma = 1
    idx = numpy.arange(n_in) 
    nonzeros = []

    for col in range(n_out):
        #shuffle and take the first 15
        numpy.random.shuffle(idx)
        nonzeros.append(list(idx[:n_nonzeros]))
    
    nonzeros = numpy.array(nonzeros)

    val = numpy.zeros((n_in, n_out), dtype='float32')

    for col in range(n_out):
        val[:,col][nonzeros[col]] = cast32(numpy.random.normal(0, sigma, size=n_nonzeros))

    val = theano.shared(value = val, name = name)
    return val


def get_shared_bias(n, name, offset = 0):
    val = numpy.zeros(n) - offset
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def dropout(IN, p = 0.5):
    noise   =   MRG.binomial(p = 1-p, n = 1, size = IN.shape, dtype='float32')
    print 'Dropout probability of masking = ', p
    #OUT     =   (IN * noise) / cast32(1 - p)
    print 'Dropout : no dividing of the outgoing weights by p --> do this at test time!'
    OUT     =   IN * noise
    return OUT

def add_gaussian_noise(IN, std = 1):
    print 'GAUSSIAN NOISE : ', std
    noise   =   MRG.normal(avg  = 0, std  = std, size = IN.shape, dtype='float32')
    OUT     =   IN + noise
    return OUT

def mul_binomial_noise(IN, p = 0.5):
    # salt and pepper? masking?
    noise   =   MRG.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
    IN      =   IN * noise
    return IN

def mul_salt_and_pepper(IN, p = 0.2):
    # salt and pepper noise
    print 'DAE uses salt and pepper noise'
    a = MRG.binomial(size=IN.shape, n=1,
                          p = 1 - p,
                          dtype='float32')
    b = MRG.binomial(size=IN.shape, n=1,
                          p = 0.5,
                          dtype='float32')
    c = T.eq(a,0) * b
    return IN * a + c

'''
mDA pseudocode


n [W,h]=mDA(X,p);

X=[X;ones(1,size(X,2))];

d=size(X,1);

q=[ones(d-1,1).*(1-p); 1];

S=X*X?;

Q=S.*(q*q?);

Q(1:d+1:end)=q.*diag(S);

P=S.*repmat(q?,d,1);

W=P(1:end-1,:)/(Q+1e-5*eye(d));

h=tanh(W*X);

'''
