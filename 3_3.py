T = int(input())
x_y = []
for i in range(T):
    x, y = map(int, input().strip().split())
    x_y.append([x, y])

# print(1985%10)
# print(1985//10)


def get_num(num):
    find_n = []
    while num > 0:
        n = num % 10
        find_n.append(n)
        num = num // 10

    return find_n

for i in range(T):
    n = x_y[i][0]
    k = x_y[i][1]
    tj_num = [j for j in range(1, n+1)]
    tj_num2 = [j for j in range(1, n+1)]
    dont_l = [w for w in range(k, 10)]

    for num in tj_num:
        find_n = get_num(num)
        for jc in dont_l:
            if jc in find_n:
                tj_num2.remove(num)
                break

    print(len(tj_num2))