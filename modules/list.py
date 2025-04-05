# prepend 'x' to list 'list'
def prepend(x, list):
    return list.insert(0, x)


def print_list(list, name):

    print(f"List {name}:")
    i=0
    for element in list:
        print(f"\telement #{i} = {element}")
        i+=1


# flatten a nested list 'list' (with an arbitrary level of flattening) and return the flattened list
def flatten_list(lst):
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten_list(item))  # Recursively flatten
        else:
            flat.append(item)
    return flat