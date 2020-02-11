import numpy as np
import torch
randomMatrix = np.random.rand(5,10);
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(
'randomMatrix', '\n' ,
 randomMatrix,
 '\n' ,
  'A', '\n' ,
   A
   )

print(
'newaxis', '\n'
  'A[:, np.newaxis]', '\n' ,
   A[:, np.newaxis],
   '\n' ,
   'A[np.newaxis, :]', '\n' ,
    A[np.newaxis, :]
   )


print(
     'A dot B' , '\n'
     , np.dot(A, B),
     '\n' ,
     'Aaxis dot Baxis' , '\n' ,
     np.dot(A[:, np.newaxis], B[np.newaxis, :])
    )


print(
     '[0.0] * 10' , '\n'
     , [0.0] * 10
    )


def num_flat_features(x):
    size = [[1,2],[3,4]]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

#print('A dot B' , '\n'
#, num_flat_features(A))


x = torch.tensor([[1, 2, 3, 4],[5,6,7,8]])
print("unsqueeze1", x.unsqueeze_(-1))

x = torch.tensor([[1, 2, 3, 4],[5,6,7,8]])
print("unsqueeze1", torch.unsqueeze(x, -1))
