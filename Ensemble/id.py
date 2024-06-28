
import pandas as pd
import os
file_path =r"Train\Faster_rcnn\Final-dataset-grad-milestone3.v20i.tensorflow\valid\_annotations.csv"
label_to_id = {0: 'Bus', 1: 'Car', 2: 'MotorCycle', 3: 'Toktok', 4:'Truck'}

df = pd.read_csv(file_path)

# Initialize a dictionary to store the file paths and their corresponding coordinates
file_data = {}
actual = []

for index, row in df.iterrows():
    file_name = row['filename']
    label = row['class']
    x_min = row['xmin']
    x_max = row['xmax']
    y_min = row['ymin']
    y_max = row['ymax']
    label_id = [key for key, value in label_to_id.items() if value == label][0]
    if actual == []:
        actual.append([file_name,[{'coordinates': [x_min, y_min, x_max, y_max], 'label_id': label_id}]])
        continue
    # Check if the file name is already in the dictionary
    if file_name not in actual[-1]:
        # If not, add it with the current coordinates
#         file_data[file_name] = [{'coordinates': [x_min, x_max, y_min, y_max], 'label_id': label_id}]
        actual.append([file_name,[{'coordinates': [x_min,y_min , x_max, y_max], 'label_id': label_id}]])

    else:
        # If it is, append the new coordinates to the existing array
#         file_data[file_name].append({'coordinates': [x_min, x_max, y_min, y_max], 'label_id': label_id})
        actual[-1][1].append({'coordinates': [x_min, y_min, x_max, y_max], 'label_id':label_id})










def convert_to_desired_format(input_list, class_labels):
    output_lines = []
    for obj in input_list[1]:
        
        label_name = class_labels.get(obj['label_id'], 'unknown')
        xmin, ymin, xmax, ymax = obj['coordinates']
        output_lines.append(f"{label_name} {xmin} {ymin} {xmax} {ymax}")
    return output_lines
        



for i in range(len(actual)):

    output_lines = convert_to_desired_format(actual[i], label_to_id)
    with open(f"mAP\\input\\ground-truth\\{os.path.splitext(actual[i][0])[0]}.txt", 'w') as file:
        for line in output_lines:
            file.write(line+'\n')




# Iterate over each row in the DataFrame
