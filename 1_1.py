T = int(input())
x_y = []
for i in range(T):
    x, y = map(float, input().strip().split())
    x_y.append([x, y])

for i in range(T):
    x, y = x_y[i][0], x_y[i][1]
    temp = " "
    if abs(x - int(x)) < 0.5 and abs(y - int(y)) < 0.5:
        x = int(x)
        y = int(y)

    elif abs(x - int(x)) < 0.5 and abs(y - int(y)) > 0.5:
        x = int(x)

        if y >= 0:
            y = int(y) + 1
        else:
            y = int(y) - 1
    elif abs(x - int(x)) > 0.5 and abs(y - int(y)) > 0.5:
        if x >= 0:
            x = int(x) + 1
        else:
            x = int(x) - 1

        if y >= 0:
            y = int(y) + 1
        else:
            y = int(y) - 1
    elif abs(x - int(x)) > 0.5 and abs(y - int(y)) < 0.5:
        if x >= 0:
            x = int(x) + 1
        else:
            x = int(x) - 1

        y = int(y)
    elif abs(x - int(x)) == 0.5 or abs(y - int(y)) == 0.5:
        if  abs(x - int(x)) <= 0.5:
            if x > 0 :
                x = int(x)
            else: x = int(x) - 1
        else :
            if x > 0:
                x = int(x) + 1
            else:
                x = int(x) -1

        if abs(y - int(y)) <= 0.5:
            if y > 0:
                y = int(y)
            else:
                y = int(y) - 1
        else:
            if y > 0:
                y = int(y) + 1
            else:
                y = int(y) - 1


    print(f"{x}{temp}{y}")
