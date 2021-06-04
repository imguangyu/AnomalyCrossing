import numpy as np
import argparse
import os

def create_Pmatrix(save_dir, M=10, matrixSize=512):
  P_matrix = []
  for i in range(M):
      A = np.random.rand(matrixSize, matrixSize)
      B = np.dot(A, A.transpose())
      C = (B + B.T) / 2
      eigenvalue, featurevector = np.linalg.eig(C)

      list_a = eigenvalue.tolist()

      list_a_min_list = min(list_a)
      min_index = list_a.index(min(list_a))
      featurevector_new = np.delete(featurevector.T, min_index, axis=0).T

      P_matrix.append(featurevector_new)

  P_matrix= np.array(P_matrix)

  if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
  save_file = os.path.join(save_dir, 'P_matrix.npy')
  np.save(save_file, P_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Pmatrix for PBSR')
    parser.add_argument('--save-dir', metavar='DIR', default='./checkpoint/PBSR/',
                        help='path to dataset files')    
    parser.add_argument('-m', '--m', default=10, type=int,
                        help='number of classification heads')
    parser.add_argument('-s', '--matrix-size', default=512, type=int, 
                        help='dimension of the features')
    args = parser.parse_args()
    create_Pmatrix(args.save_dir, args.m, args.matrix_size)