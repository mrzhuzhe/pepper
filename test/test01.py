import numpy as np
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
