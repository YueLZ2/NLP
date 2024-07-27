T = int(input())
x_y = []
for i in range(T):
    q = input().strip().split()
    x_y.append((q[0], float(q[1])))

for i in range(T):
    if x_y[i][0] == "dice":
        output_1 = x_y[i][1] /(2 - x_y[i][1])
    else:
        output_1 = x_y[i][1] * 2 /(x_y[i][1]+1)
    print('%.2f'%output_1)
