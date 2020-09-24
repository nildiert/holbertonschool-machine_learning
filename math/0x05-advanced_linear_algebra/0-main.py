#!/usr/bin/env python3

if __name__ == '__main__':
    determinant = __import__('0-determinant').determinant

    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]
    mat7 = [[3, 2, -1, 4], [2, 1, 5, 7], [0, 5, 2, -6], [-1, 2, 1, 0]]
    mat8 = ""
    mat9 = [[1, 2, 3, 4, 5, 9, 2, 5, 6, 7],
            [6, 7, 8, 9, 1, 5, 4, 2, 9, 7],
            [7, 8, 9, 1, 2, 5, 4, 3, 1, 2],
            [3, 4, 8, 6, 7, 5, 4, 3 , 5, 1],
            [3, 4, 5, 9, 3, 5, 4, 4 , 5, 1],
            [3, 2, 8, 6, 9, 5, 4, 3 , 5, 1],
            [3, 4, 5, 6, 6, 9, 4, 3 , 5, 1],
            [3, 6, 8, 6, 7, 5, 9, 2 , 5, 1],
            [3, 4, 5, 6, 8, 5, 4, 9 , 5, 1],
            [3, 4, 8, 6, 7, 5, 4, 3 , 9, 1]]
    mat10 = [[5, 7, 9, 8], (3, 1, 8, 5), [6, 2, 4, 1], [1, 2, 3, 4]]

    print(determinant(mat0))
    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat4))
    print(determinant(mat7))
    print(determinant(mat9))
    try:
        determinant(mat10)
    except Exception as e:
        print(e)
    try:
        determinant(mat5)
    except Exception as e:
        print(e)
    try:
        determinant(mat6)
    except Exception as e:
        print(e)

    try:
        determinant(mat8)
    except Exception as e:
        print(e)

