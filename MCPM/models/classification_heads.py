from .utils import euclidean_metric
import torch
import torch.nn as nn
from qpth.qp import QPFunction
from torch.autograd import Variable
from torch.nn.utils.weight_norm import WeightNorm
import models
from utils.model_path import rgb_3d_model_path_selection


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))

def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.
    
    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """
    # torch.gesv is not supported any more, move to torch.solve
    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).cuda()
    b_inv, _ = torch.solve(id_matrix, b_mat)
    # b_inv, _ = torch.gesv(id_matrix, b_mat)
    
    return b_inv

def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def MetaOptNetHead_SVM_CS(query, support, support_labels, n_way, n_shot, C_reg=0.1, double_precision=False, maxIter=15):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).

    This model is the classification head that we use for the final version.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    #and C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    #This borrows the notation of liblinear.
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)

    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
    #This seems to help avoid PSD error from the QP solver.
    block_kernel_matrix += 1.0 * torch.eye(n_way*n_support).expand(tasks_per_batch, n_way*n_support, n_way*n_support).cuda()
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)
    
    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot
    #print (G.size())
    #This part is for the inequality constraints:
    #\alpha^m_i <= C^m_i \forall m,i
    #where C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
    C = Variable(id_matrix_1)
    h = Variable(C_reg * support_labels_one_hot)
    #print (C.size(), h.size())
    #This part is for the equality constraints:
    #\sum_m \alpha^m_i=0 \forall i
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

    A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
    b = Variable(torch.zeros(tasks_per_batch, n_support))
    #print (A.size(), b.size())
    if double_precision:
        G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits

def MetaOptNetHead_SVM_He(query, support, support_labels, n_way, n_shot, C_reg=0.01, double_precision=False):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    A simplified multi-class support vector machine with reduced dual optimization
    (He et al., Pattern Recognition Letter 2012).
    
    This SVM is desirable because the dual variable of size is n_support
    (as opposed to n_way*n_support in the Weston&Watkins or Crammer&Singer multi-class SVM).
    This model is the classification head that we have initially used for our project.
    This was dropped since it turned out that it performs suboptimally on the meta-learning scenarios.
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    
    kernel_matrix = computeGramMatrix(support, support)
    #There will be an error is use support_labels the dim doesn't match
    #So I think that here we should use one hot version of the labels 
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # V = (support_labels * n_way - torch.ones(tasks_per_batch, n_support, n_way).cuda()) / (n_way - 1)
    V = (support_labels_one_hot * n_way - torch.ones(tasks_per_batch, n_support, n_way).cuda()) / (n_way - 1)
    G = computeGramMatrix(V, V).detach()
    G = kernel_matrix * G
    
    e = Variable(-1.0 * torch.ones(tasks_per_batch, n_support))
    id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support)
    C = Variable(torch.cat((id_matrix, -id_matrix), 1))
    h = Variable(torch.cat((C_reg * torch.ones(tasks_per_batch, n_support), torch.zeros(tasks_per_batch, n_support)), 1))
    dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.

    if double_precision:
        G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]
    else:
        G, e, C, h = [x.cuda() for x in [G, e, C, h]]
        
    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(query, support)
    compatibility = compatibility.float()

    logits = qp_sol.float().unsqueeze(1).expand(tasks_per_batch, n_query, n_support)
    logits = logits * compatibility
    logits = logits.view(tasks_per_batch, n_query, n_shot, n_way)
    logits = torch.sum(logits, 2)

    return logits

def MetaOptNetHead_SVM_WW(query, support, support_labels, n_way, n_shot, C_reg=0.00001, double_precision=False):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    Support Vector Machines for Multi Class Pattern Recognition
    (Weston and Watkins, ESANN 1999).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    Support Vector Machines for Multi Class Pattern Recognition
    (Weston and Watkins, ESANN 1999).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #In theory, \alpha is an (n_support, n_way) matrix
    #NOTE: In this implementation, we solve for a flattened vector of size (n_way*n_support)
    #In order to turn it into a matrix, you must first reshape it into an (n_way, n_support) matrix
    #then transpose it, resulting in (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support) + torch.ones(tasks_per_batch, n_support, n_support).cuda()
    
    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
    block_kernel_matrix = batched_kronecker(id_matrix_0, kernel_matrix)
    
    kernel_matrix_mask_x = support_labels.reshape(tasks_per_batch, n_support, 1).expand(tasks_per_batch, n_support, n_support)
    kernel_matrix_mask_y = support_labels.reshape(tasks_per_batch, 1, n_support).expand(tasks_per_batch, n_support, n_support)
    kernel_matrix_mask = (kernel_matrix_mask_x == kernel_matrix_mask_y).float()
    
    block_kernel_matrix_inter = kernel_matrix_mask * kernel_matrix
    block_kernel_matrix += block_kernel_matrix_inter.repeat(1, n_way, n_way)
    
    kernel_matrix_mask_second_term = support_labels.reshape(tasks_per_batch, n_support, 1).expand(tasks_per_batch, n_support, n_support * n_way)
    kernel_matrix_mask_second_term = kernel_matrix_mask_second_term == torch.arange(n_way).long().repeat(n_support).reshape(n_support, n_way).transpose(1, 0).reshape(1, -1).repeat(n_support, 1).cuda()
    kernel_matrix_mask_second_term = kernel_matrix_mask_second_term.float()
    
    block_kernel_matrix -= (2.0 - 1e-4) * (kernel_matrix_mask_second_term * kernel_matrix.repeat(1, 1, n_way)).repeat(1, n_way, 1)

    Y_support = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    Y_support = Y_support.view(tasks_per_batch, n_support, n_way)
    Y_support = Y_support.transpose(1, 2)   # (tasks_per_batch, n_way, n_support)
    Y_support = Y_support.reshape(tasks_per_batch, n_way * n_support)
    
    G = block_kernel_matrix

    e = -2.0 * torch.ones(tasks_per_batch, n_way * n_support)
    id_matrix = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
            
    C_mat = C_reg * torch.ones(tasks_per_batch, n_way * n_support).cuda() - C_reg * Y_support

    C = Variable(torch.cat((id_matrix, -id_matrix), 1))
    #C = Variable(torch.cat((id_matrix_masked, -id_matrix_masked), 1))
    zer = torch.zeros(tasks_per_batch, n_way * n_support).cuda()
    
    h = Variable(torch.cat((C_mat, zer), 1))
    
    dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.

    if double_precision:
        G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]
    else:
        G, e, C, h = [x.cuda() for x in [G, e, C, h]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    #qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
    qp_sol = QPFunction(verbose=False)(G, e, C, h, dummy.detach(), dummy.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query) + torch.ones(tasks_per_batch, n_support, n_query).cuda()
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(1).expand(tasks_per_batch, n_way, n_support, n_query)
    qp_sol = qp_sol.float()
    qp_sol = qp_sol.reshape(tasks_per_batch, n_way, n_support)
    A_i = torch.sum(qp_sol, 1)   # (tasks_per_batch, n_support)
    A_i = A_i.unsqueeze(1).expand(tasks_per_batch, n_way, n_support)
    qp_sol = qp_sol.float().unsqueeze(3).expand(tasks_per_batch, n_way, n_support, n_query)
    Y_support_reshaped = Y_support.reshape(tasks_per_batch, n_way, n_support)
    Y_support_reshaped = A_i * Y_support_reshaped
    Y_support_reshaped = Y_support_reshaped.unsqueeze(3).expand(tasks_per_batch, n_way, n_support, n_query)
    logits = (Y_support_reshaped - qp_sol) * compatibility

    logits = torch.sum(logits, 2)

    return logits.transpose(1, 2)

def MetaOptNetHead_Ridge(query, support, support_labels, n_way, n_shot, lambda_reg=50.0, double_precision=False):
    """
    Fits the support set with ridge regression and 
    returns the classification score on the query set.

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      lambda_reg: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)
    kernel_matrix += lambda_reg * torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

    block_kernel_matrix = kernel_matrix.repeat(n_way, 1, 1) #(n_way * tasks_per_batch, n_support, n_support)
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.transpose(0, 1) # (n_way, tasks_per_batch * n_support)
    support_labels_one_hot = support_labels_one_hot.reshape(n_way * tasks_per_batch, n_support)     # (n_way*tasks_per_batch, n_support)
    
    G = block_kernel_matrix
    e = -2.0 * support_labels_one_hot
    
    #This is a fake inequlity constraint as qpth does not support QP without an inequality constraint.
    id_matrix_1 = torch.zeros(tasks_per_batch*n_way, n_support, n_support)
    C = Variable(id_matrix_1)
    h = Variable(torch.zeros((tasks_per_batch*n_way, n_support)))
    dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.

    if double_precision:
        G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]

    else:
        G, e, C, h = [x.float().cuda() for x in [G, e, C, h]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
    #qp_sol = QPFunction(verbose=False)(G, e.detach(), dummy.detach(), dummy.detach(), dummy.detach(), dummy.detach())

    #qp_sol (n_way*tasks_per_batch, n_support)
    qp_sol = qp_sol.reshape(n_way, tasks_per_batch, n_support)
    #qp_sol (n_way, tasks_per_batch, n_support)
    qp_sol = qp_sol.permute(1, 2, 0)
    #qp_sol (tasks_per_batch, n_support, n_way)
    
    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits

def MetaOptNetHead_Ridge_CS(query, support, support_labels, n_way, n_shot, lambda_reg=1.0):
    """
    Fits the support set with ridge regression and 
    returns the classification score on the query set.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      lambda_reg: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    # pdb.set_trace()
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0)
           and query.size(2) == support.size(2))
    # n_support must equal to n_way * n_shot
    assert(n_support == n_way * n_shot)

    support_labels_one_hot = one_hot(support_labels.view(
        tasks_per_batch * n_support), n_way).view(tasks_per_batch, n_support, n_way)

    logits = []
    train_loss = []
    for i in range(tasks_per_batch):
        query_i = query[i]
        support_i = support[i]
        support_labels_i = support_labels_one_hot[i]

        close_w = torch.mm(torch.mm(torch.transpose(support_i, 0, 1), torch.inverse(torch.mm(support_i, torch.transpose(
            support_i, 0, 1)) + lambda_reg*torch.eye(n_support).cuda())), support_labels_i)

        # pdb.set_trace()
        train_log = torch.mm(support_i, close_w)
        train_loss.append(torch.mean((train_log - support_labels_i)**2)*0.5)

        logits_q = torch.mm(query_i, close_w)
        logits.append(logits_q)

    logits = torch.stack(logits)

    return logits

def R2D2Head(query, support, support_labels, n_way, n_shot, l2_regularizer_lambda=50.0):
    """
    Fits the support set with ridge regression and 
    returns the classification score on the query set.
    
    This model is the classification head described in:
    Meta-learning with differentiable closed-form solvers
    (Bertinetto et al., in submission to NIPS 2018).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      l2_regularizer_lambda: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()
    
    # Compute the dual form solution of the ridge regression.
    # W = X^T(X X^T - lambda * I)^(-1) Y
    ridge_sol = computeGramMatrix(support, support) + l2_regularizer_lambda * id_matrix
    ridge_sol = binv(ridge_sol)
    ridge_sol = torch.bmm(support.transpose(1,2), ridge_sol)
    ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)
    
    # Compute the classification score.
    # score = W X
    logits = torch.bmm(query, ridge_sol)

    return logits

def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and 
    returns the classification score (=L2 distance to each class prototype) on the query set.
    
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)
    
    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    #************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1,2)
    # Batch matrix multiplication:
    #   prototypes = labels_train_transposed * features_train ==>
    #   [batch_size x nKnovel x num_channels] =
    #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits
    
    if normalize:
        logits = logits / d

    return logits

class cosineDist(nn.Module):
    def __init__(self, indim, outdim):
        super(cosineDist, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.
    
    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores
class ClassificationHead(nn.Module):
    def __init__(self, base_learner='SVM-CS', n_shot=5, n_way=5, enable_scale=True):
        super(ClassificationHead, self).__init__()
        if ('SVM-CS' in base_learner):
            self.head = MetaOptNetHead_SVM_CS
        elif ('RidgeCS' in base_learner):
            self.head = MetaOptNetHead_Ridge_CS
        elif ('Ridge' in base_learner):
            self.head = MetaOptNetHead_Ridge
        elif ('R2D2' in base_learner):
            self.head = R2D2Head
        elif ('Proto' in base_learner):
            self.head = ProtoNetHead
        elif ('SVM-He' in base_learner):
            self.head = MetaOptNetHead_SVM_He
        elif ('SVM-WW' in base_learner):
            self.head = MetaOptNetHead_SVM_WW
        else:
            print ("Cannot recognize the base learner type")
            assert(False)
        
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.n_way = n_way
        self.n_shot = n_shot
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
    
    def forward_bsr(self, support, query, support_labels, **kwargs):
        query = self.avgpool(query).squeeze().unsqueeze(dim=0)
        # in case of batch size == 1
        if query.dim() < 3:
            query = query.unsqueeze(dim=0)
        support = self.avgpool(support).squeeze()
        u, s, v = torch.svd(support.t())
        bsr = torch.sum(torch.pow(s, 2))

        support = support.unsqueeze(dim=0)
        # in case of batch size == 1
        if support.dim() < 3:
            support = support.unsqueeze(dim=0)
        support_labels = support_labels.unsqueeze(dim=0)

        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels,  self.n_way,  self.n_shot, **kwargs).squeeze(0), bsr
        else:
            return self.head(query, support, support_labels,  self.n_way,  self.n_shot, **kwargs).squeeze(0), bsr

    def forward(self, support, query, support_labels, **kwargs):
        query = self.avgpool(query).squeeze().unsqueeze(dim=0)
        # in case of batch size == 1
        if query.dim() < 3:
            query = query.unsqueeze(dim=0)
        support = self.avgpool(support).squeeze().unsqueeze(dim=0)
        # in case of batch size == 1
        if support.dim() < 3:
            support = support.unsqueeze(dim=0)
        support_labels = support_labels.unsqueeze(dim=0)
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels,  self.n_way,  self.n_shot, **kwargs).squeeze(0)
        else:
            return self.head(query, support, support_labels,  self.n_way,  self.n_shot, **kwargs).squeeze(0)
    
    def set_forward_feature(self, support, query, support_labels, **kwargs):
        query= query.unsqueeze(dim=0)
        support = support.unsqueeze(dim=0)
        support_labels = support_labels.unsqueeze(dim=0)
        logits_query = self.head(query, support, support_labels,  self.n_way,  self.n_shot, **kwargs).squeeze(0)
        logits_support = self.head(support, support, support_labels,  self.n_way,  self.n_shot, **kwargs).squeeze(0)
        return logits_query, logits_support

class protonet(nn.Module):
    def __init__(self, shot, way, temperature):
        super(protonet, self).__init__()
        self.shot = shot
        self.way = way 
        self.temperature = temperature
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

    def forward(self, support, query, support_labels=None):
        proto = self.avgpool(support)
        proto = proto.contiguous().view(self.shot, self.way, -1).mean(dim=0)
        query = self.avgpool(query)
        query = query.squeeze()
        # in case of batch size == 1
        if query.dim() < 2:
            query = query.unsqueeze(0)
        logits = euclidean_metric(query, proto) / self.temperature
        return logits
    
    def forward_bsr(self, support, query, support_labels=None,):
        proto = self.avgpool(support)
        u, s, v = torch.svd(proto.squeeze().t())
        bsr = torch.sum(torch.pow(s, 2))
        proto = proto.contiguous().view(self.shot, self.way, -1).mean(dim=0)
        query = self.avgpool(query)
        query = query.squeeze()
        # in case of batch size == 1
        if query.dim() < 2:
            query = query.unsqueeze(0)
        logits = euclidean_metric(query, proto) / self.temperature
        return logits, bsr

    def set_forward_feature(self, support, query, y_support=None):
        proto = support.contiguous().view(self.shot, self.way, -1).mean(dim=0)
        logits_query = euclidean_metric(query, proto) / self.temperature
        logits_support = euclidean_metric(support, proto) / self.temperature

        return logits_query, logits_support

class BSR_Single(nn.Module):
    def __init__(self, feat_dim, n_way, lamda=0.001):
        super(BSR_Single, self).__init__()
        self.feat_dim = feat_dim
        self.classifier = nn.Linear(self.feat_dim, n_way)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.n_way = n_way
        # self.n_shot = n_shot
        self.loss_fn = nn.CrossEntropyLoss()
        self.lamda = lamda

    def forward(self, x):
        feature = self.avgpool(x).squeeze()
        # in case of batch size == 1
        if feature.dim() < 2:
            feature.unsqueeze(0)
        u, s, v = torch.svd(feature.t())
        bsr = torch.sum(torch.pow(s, 2))
        scores = self.classifier(feature)
        return scores, bsr

    def forward_loss(self, x, y, label_smoothing=False, eps=0.1):
        scores, bsr = self.forward(x)
        if label_smoothing:
            smoothed_y = one_hot(y.reshape(-1), self.n_way) 
            log_prb = F.log_softmax(scores.reshape(-1, self.n_way), dim=1)
            loss_c = -(smoothed_y * log_prb).sum(dim=1)
            loss_c = loss_c.mean()
        else: 
            loss_c = self.loss_fn(scores, y)
        loss = loss_c + self.lamda * bsr
        return loss, scores

class PBSR_Ensemble(nn.Module):
    def __init__(self, arch, feat_dim, n_way, P_matrix, lamda=0.001, multi_gpu=True):
        super(PBSR_Ensemble, self).__init__()
        self.encoder = models.__dict__[arch](modelPath=rgb_3d_model_path_selection(arch))
        self.encoder_without_dp = self.encoder
        if multi_gpu and torch.cuda.device_count() > 1:
            self.encoder=torch.nn.DataParallel(self.encoder)
            self.encoder_without_dp = self.encoder.module

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.classifier = nn.Linear(feat_dim - 1, n_way)
        self.classifier.bias.data.fill_(0)

        self.n_way = n_way
        self.loss_fn = nn.CrossEntropyLoss()

        self.P_matrix = P_matrix
        self.lamda = lamda

    def forward(self, x):
        fea_b = self.encoder(x)
        fea_b = self.avgpool(fea_b).squeeze()
        # in case of batch size == 1
        if fea_b.dim() < 2:
            fea_b.unsqueeze(0)
        fea_e = torch.mm(fea_b, self.P_matrix)
        u, s, v = torch.svd(fea_e.t())
        bsr = torch.sum(torch.pow(s, 2))
        scores = self.classifier(fea_e)
        return scores, bsr

    def forward_loss(self, x, y):
        scores, bsr = self.forward(x)
        loss_c = self.loss_fn(scores, y)
        loss = loss_c + self.lamda * bsr
        return loss

# share the same encoder
class PBSR_Share(nn.Module):
    def __init__(self, feat_dim, n_way, P_matrix, lamda=0.001):
        super(PBSR_Share, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.classifier = nn.Linear(feat_dim - 1, n_way)
        self.classifier.bias.data.fill_(0)

        self.n_way = n_way
        self.loss_fn = nn.CrossEntropyLoss()

        self.P_matrix = P_matrix
        self.lamda = lamda

    def forward(self, x):
        fea_b = self.avgpool(x).squeeze()
        # in case of batch size == 1
        if fea_b.dim() < 2:
            fea_b.unsqueeze(0)
        fea_e = torch.mm(fea_b, self.P_matrix)
        u, s, v = torch.svd(fea_e.t())
        bsr = torch.sum(torch.pow(s, 2))
        scores = self.classifier(fea_e)
        return scores, bsr

    def forward_loss(self, x, y):
        scores, bsr = self.forward(x)
        loss_c = self.loss_fn(scores, y)
        loss = loss_c + self.lamda * bsr
        return loss