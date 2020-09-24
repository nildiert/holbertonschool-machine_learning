#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)


vals = np.arange(0, 110, 10)
plt.hist(student_grades, edgecolor="black", bins=vals)
plt.xticks(np.arange(0, 110, step=10))
plt.yticks(np.arange(0, 35, 5))
plt.xlim(0, 100)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
