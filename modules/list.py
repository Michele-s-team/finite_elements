# prepend 'x' to list 'list'
def prepend(x, list):
    return list.insert(0, x)


def print_list(list, name):

    print(f"List {name}:")
    i=0
    for element in list:
        print(f"\telement #{i} = {element}")
        i+=1
