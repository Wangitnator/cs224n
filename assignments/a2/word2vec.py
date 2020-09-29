#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1/(1+np.exp(-x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    ### YOUR CODE HERE
    X = outsideVectors.dot( centerWordVec ) # (N x 1) == (N x d) * (d x 1)

    Y_hat = softmax(X)
    Y_hat = Y_hat.reshape((Y_hat.shape[0], 1))# Should be (N x 1)

    Y_hat_transpose = np.transpose(Y_hat)
    y_hat = Y_hat[outsideWordIdx] # scalar

    loss = -np.log(y_hat)

    gradCenterVec = ( # Should be (1 x d)
            -outsideVectors[outsideWordIdx, :] # should be (1 x d)
            + Y_hat_transpose.dot( outsideVectors ) # should be (d x d) == (d x N) * (N x d)
    )
    v_c = centerWordVec.reshape((centerWordVec.shape[0],1))
    v_c_transpose = v_c.transpose()

    gradOutsideVecs = np.dot(Y_hat, v_c_transpose) # (N x 1) * (1 x d)
    gradOutsideVecs[outsideWordIdx, :] = -v_c_transpose*(1 - y_hat) # (1 x d)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 


    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE
    u_o_transpose = outsideVectors[outsideWordIdx, :] # is (1 x d)
    u_o = np.transpose(u_o_transpose) # is (d x 1)

    loss_t1 = -np.log(sigmoid( u_o_transpose.dot( centerWordVec ) ) ) # is (1 x 1) == (1 x d) * (d x 1)
    loss_t2 = -np.sum( # is (1 x 1)
        np.log( # is (K x 1)
            sigmoid( # is (K x 1)
                -outsideVectors[negSampleWordIndices,:].dot( centerWordVec ) # is (K x d) * (d x 1)
            )
        ),
        axis=0
    )

    loss = loss_t1 + loss_t2

    gradCenterVec_t1 = -  u_o * ( # is (1 x 1)
        1 - sigmoid( # (1 x 1)
            u_o_transpose.dot( centerWordVec ) # (1 x 1) == (1 x d) * (d x 1)
        )
    )

    u_k_transpose = -outsideVectors[negSampleWordIndices, :] # is (K x d)
    u_k = np.transpose(u_k_transpose)

    gradCenterVec_t2 = -u_k.dot(
        1
        - sigmoid(u_k_transpose.dot(centerWordVec))
    ) # # (d x K) * (K x d) * (d x 1)

    '''( # is is (d X 1) == (d x K) * (K x d) * (d x 1)
            -outsideVectors[negSampleWordIndices, :] # is (K x d)
            * sigmoid( # is (K x d)
                np.transpose( # is (K x d)
                    -outsideVectors[:,negSampleWordIndices])  # is (d x K)
                * centerWordVec # is (d x 1)
            )
    )'''

    gradCenterVec = gradCenterVec_t1 + gradCenterVec_t2

    u_o_transpose = u_o_transpose.reshape((1,u_o_transpose.shape[0]))
    centerWordVec2 = centerWordVec.reshape((centerWordVec.shape[0],1))
    z = sigmoid(u_o_transpose.dot(centerWordVec2)) - 1
    gradOutsideVecs = np.zeros(outsideVectors.shape)

    np.dot(outsideVectors, centerWordVec2)

    gradOutsideVec = centerWordVec2 * z # is (1 x 1) = (1 x d) * (d x 1)
    #gradOutsideVecs += -(sigmoid(np.dot(outsideVectors, centerWordVec)) - 1).dot(centerWordVec.transpose())
    gradOutsideVecs[outsideWordIdx,:] += gradOutsideVec.reshape((gradOutsideVec.shape[0],))

    u_k = outsideVectors[negSampleWordIndices]
    z = sigmoid(-np.dot(u_k, centerWordVec2))

    #loss += np.sum(- np.log(z))
    #gradCenterVecNeg = np.dot((z - 1).transpose(), u_k).transpose()
    #gradCenterVec += gradCenterVecNeg.reshape((gradCenterVecNeg.shape[0],))
    #gradOutsideVecs[negSampleWordIndices] += np.outer((z-1),centerWordVec)*(-1)
    '''
    for k in range(K):
        samp = indices[k+1]
        z = sigmoid(np.dot(-outsideVectors[samp], centerWordVec))
        gradOutsideVecs[samp] -=  centerWordVec * (z - 1.0)
    '''

    ### Please use your implementation of sigmoid in here.


    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE
    centerWordIndex = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[centerWordIndex, :]
    outsideWordIndices = [word2Ind[index] for index in outsideWords] # is at most (2 * windowSize) elements

    # word vectors of all outside window words
    for outsideWordIndex in outsideWordIndices:
        windowWordLoss, windowWordGradCenterVec, windowWordGradOutsideVecs = \
            word2vecLossAndGradient(centerWordVec, outsideWordIndex, outsideVectors, dataset)
        loss += windowWordLoss

        gradCenterVecs[centerWordIndex, :] += windowWordGradCenterVec.reshape((np.max(windowWordGradCenterVec.shape),))
        #print(f"windowWordGradOutsideVecs.shape={windowWordGradOutsideVecs.shape}")
        #print(f"gradOutsideVectors.shape={gradOutsideVectors.shape}")
        #print(f"windowWordGradOutsideVecs={windowWordGradOutsideVecs}")

        gradOutsideVectors += windowWordGradOutsideVecs

    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVectors

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")
    print ("Skip-Gram with naiveSoftmaxLossAndGradient")

    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset) 
        )
    )

    print ("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print ("Skip-Gram with negSamplingLossAndGradient")   
    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
            dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)
        )
    )
    print ("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)

if __name__ == "__main__":
    test_word2vec()
