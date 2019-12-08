import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2.5, 4.3, 6.9, 8.2, 11, 12.2, 13])

m = 1.9783513979854612
c = 0.22736111111960952

def error(m, c, x, y):
    # M and C are broadcasted into x
    estimate_y = m * x + c
    
    error = np.mean((estimate_y - y) ** 2)
    #print(estimate_y)
    print(error)

def update(m, c, x, y):
    estimate_y = m * x + c
    gradient_m = np.mean(2 * (y - estimate_y) * -x)

    m_new = m - gradient_m * 0.04

    gradient_c = np.mean(-2 * (y - estimate_y))
    c_new = c - gradient_c * 0.04

    #print(gradient)
    
    return m_new, c_new
    

error(m, c, x, y)

for i in range(500):
    m, c = update(m, c, x, y)

print(error(m, c, x, y))
print(m, c)

print(np.polyfit(x, y, 7))