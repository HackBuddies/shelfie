from scipy import misc


def euclidean(pixel1, pixel2):
    return sum([(int(pixel1[i]) - int(pixel2[i])) ** 2 for i in range(3)])


def find(small, large, dist_func):
    if len(small) == 0 or len(large) == 0 or len(small[0]) == 0 or len(large[0]) == 0:
        raise Exception("Bruh, empty images")
    min_diff = 255 ** 2 * 3 + 255
    min_r_start = min_c_start = max(len(large), len(large[0])) + 1
    for row_start in range(len(large) - len(small)):
        for col_start in range(len(large[0]) - len(small[0])):
            print("% done", row_start * col_startn)
            diff = 0
            for i in range(len(small)):
                for j in range(len(small[0])):
                    diff += dist_func(small[i][j], large[i][j])
            if diff < min_diff:
                min_r_start = row_start
                min_c_start = col_start

    return min_r_start, min_c_start


def main():
    label = misc.imread("images/label.jpg")
    test = misc.imread("images/test.jpg")
    print(find(small=label, large=test, dist_func=euclidean))


if __name__ == '__main__':
    main()
