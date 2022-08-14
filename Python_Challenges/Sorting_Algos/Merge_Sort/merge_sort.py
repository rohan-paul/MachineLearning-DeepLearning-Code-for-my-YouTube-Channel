def merge(left, right):
    new_list = []

    while min(len(left), len(right)) > 0:
        # Now sorting between left[0] & right[0]
        # and removing the lesser item from the
        # corresponding list
        if left[0] < right[0]:
            new_list.append(left.pop(0))
        else:
            new_list.append(right.pop(0))

    return new_list + left + right


def merge_sort(list):
    if len(list) == 1:
        return list

    return merge(
        merge_sort(list[: len(list) // 2]),
        merge_sort(list[len(list) // 2 :]),
    )


list = [4, 1, 3, 2, 6, 3, 18, 2, 9, 7, 3, 1, 2.5, -9]

new_list = merge_sort(list)

print(new_list)
