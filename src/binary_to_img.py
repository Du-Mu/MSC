import subprocess
import numpy
import scipy.misc
import os
import array
import glob

def get_image(input_dir, output_dir, filename):
    out_path = os.path.splitext(os.path.basename(filename))[0] + '.png'
    out_file_full = output_dir + out_path
    input_file_path = os.path.join(input_dir, filename)
    print("out_file_full: ", out_file_full)
    if os.access(filename, os.X_OK):
        f = open(input_file_path, 'rb')
        file_size = os.path.getsize(input_file_path)  
        width = 256
        rem = file_size % width

        a = array.array("B") 
        a.fromfile(f, file_size - rem)
        f.close()
        g = numpy.reshape(a, (len(a) // width, width))
        g = numpy.uint8(g)
        scipy.misc.imsave(out_file_full, g)  
    else:
        print("not exe")

def convert_bin_to_img(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(input_dir, 'Input directory not found. Exiting.')
        exit(0)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    files = os.listdir(input_dir)
    print(files)
    for filename in files:
        print("filename: ", filename)
        try:
            get_image(input_dir, output_dir, filename)
            print('Complete ', filename)
        except:
             print('Ignoring ', filename)