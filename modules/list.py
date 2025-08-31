import csv
import os

# prepend 'x' to list 'list'
def prepend(x, list):
    return list.insert(0, x)


def print_list(list, name):

    print(f"List {name}:")
    i=0
    for element in list:
        print(f"\telement #{i} = {element}")
        i+=1


'''
print a list to a csv file in a single row
Input values: 
- 'list': the list
- 'file_path': the path (path + filename + file extension) and  of the file where the list will be written. If the folder in file_path does not exist, it will be created.
'''
def print_to_csv_file(list, file_path):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list)  # Writes as one row


# flatten a nested list 'list' (with an arbitrary level of flattening) and return the flattened list
def flatten_list(lst):
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten_list(item))  # Recursively flatten
        else:
            flat.append(item)
    return flat