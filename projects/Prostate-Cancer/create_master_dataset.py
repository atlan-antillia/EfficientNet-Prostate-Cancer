# Copyright 2022 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


#
# create_master_dataset.py
#
#
# 2022/09/16 Copyright (C) antillia.com

# Please download ocular-disease-recognition-odir5k dataset from the following website:
#
# Ocular Disease Recognition
#
# https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k


# This script will split the preprocessed_images into labeled folders of
#  labels = ['N', 'D', 'G','C','A', 'M', 'O'] by using full_df.csv file.
#


import os
import sys
import csv
import shutil
import glob
import traceback


# classes 
"""
Normal (N),
Diabetes (D),
Glaucoma (G),
Cataract (C),
Age related Macular Degeneration (A),
Hypertension (H),
Pathological Myopia (M),
Other diseases/abnormalities (O)

"""

# data_label_csv


def classify(data_label_csv, images_dir, output_dir):
  classes = ['N',
             'D',
             'G',
             'C',
             'A',
             'M',
             'O']
  with open(data_label_csv, encoding='utf8', newline='') as f:
    csvreader = csv.reader(f)
    header    = next(csvreader)
    print(header)
    print(" header len {}".format(len(header)))

    for row in csvreader:
       
       filename = row[18]
       label    = row[16]
       label    = label.replace('[', '').replace(']', '')
       label    = label.replace("'","")

       print("--- filename {}  label {}".format(filename,label))

       classified_output_dir = os.path.join(output_dir, label)

       if not os.path.exists(classified_output_dir):        
         os.makedirs(classified_output_dir)
       image_file_path = os.path.join(images_dir, filename)
       if os.path.exists(image_file_path):
         shutil.copy2(image_file_path, classified_output_dir)
         print("--- copied {} to {}".format(image_file_path, classified_output_dir))
       else:
         print("Error: Not found {}".format(image_file_path))

       #print(row)    


if __name__ == '__main__':
  try:
    data_label_csv = './full_df.csv'
    images_dir     = './preprocessed_images'
    output_dir     = './Ocular_Disease_master'
    
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    classify(data_label_csv, images_dir, output_dir)

  except:
    traceback.print_exc()
