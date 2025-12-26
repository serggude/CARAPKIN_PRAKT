def homework_advanced(image_pixels, filter_size, iterations):
    height = len(image_pixels)
    width = len(image_pixels[0])
    radius = filter_size // 2

    current = [row[:] for row in image_pixels]

    for _ in range(iterations):
        next_image = [row[:] for row in current]

        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                neighborhood = []

                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        neighborhood.append(current[i + di][j + dj])

                neighborhood.sort()
                next_image[i][j] = neighborhood[len(neighborhood) // 2]

        current = next_image

    return current

def print_image(img, title):
    print(title)
    for row in img:
        print(row)
    print()


test_image = [
    [10, 10, 10, 10, 10, 10, 10],
    [10, 50, 50, 50, 50, 50, 10],
    [10, 50,  0,  0,  0, 50, 10],
    [10, 50,  0,100,  0, 50, 10],
    [10, 50,  0,  0,  0, 50, 10],
    [10, 50, 50, 50, 50, 50, 10],
    [10, 10, 10, 10, 10, 10, 10],
]

print_image(test_image, "Исходное изображение:")

filter_size = 3
iterations = 1

filtered_image = homework_advanced(test_image, filter_size, iterations)

print_image(filtered_image, "После медианного фильтра:")


